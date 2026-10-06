// CUDA kernels and device context for the complex CG solver gscg_zwf (SALMON_CG_CUDA).
// The 2x2 ZHEGV of each orbital and the step decisions stay on the CPU.
#include <cuda_runtime.h>
#include <cstddef>
#include <limits>
#include <new>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>
#include <type_traits>
#ifdef SALMON_CG_ORDERED_NONLOCAL
#include "cg_stencil_ordered.h"
#include "cg_nonlocal_ordered.h"
#endif

// Interoperable with epoch_input in cg_resident_bridge.f90. Host pointers are
// borrowed only during begin; Context owns separate copies until epoch close.
struct EpochInput {
    int lo[3],hi[3],no,nk,nlma,nps,natom,max_vi,max_contrib;
    const int *idx,*idy,*idz;
    const double *vlocal,*lap0,*lapt,*nabt;
    const int *ia_tbl,*mps,*jxyz;
    const double2 *uv;
    const double *rinv;
    const int *v2j,*v2nlma,*k2ilma,*k2j;
};
struct Grid {
    int nx,ny,nz,halo,nv;
    size_t sx,sy,plane,field;
    __device__ size_t at(int ix,int iy,int iz,int v) const {
        return size_t(v)*field+size_t(iz+halo)*plane+size_t(iy+halo)*sx+ix+halo;
    }
};
__device__ __forceinline__ double2 sum(double2 a,double2 b){
    return make_double2(__dadd_rn(a.x,b.x),__dadd_rn(a.y,b.y));
}
// Same arithmetic as the CPU loops of gscg_zwf compiled by nvfortran 26.1: complex products
// use FMA, and grid points are summed one at a time in loop order.
__device__ __forceinline__ double2 conjugate_product(double2 a,double2 b){
    return make_double2(__fma_rn(a.x,b.x,__dmul_rn(a.y,b.y)),
                        __fma_rn(a.x,b.y,__dmul_rn(-a.y,b.x)));
}
__device__ __forceinline__ double2 multiply(double2 c,double2 a){
    return make_double2(__fma_rn(c.x,a.x,__dmul_rn(-c.y,a.y)),
                        __fma_rn(c.x,a.y,__dmul_rn(c.y,a.x)));
}
__device__ __forceinline__ double2 updated_gradient(double2 x,double2 p,double2 &hx,
        double2 hp,double2 u1,double2 u2,double e){
    hx=sum(multiply(u1,hx),multiply(u2,hp));
    const double2 proposed=sum(multiply(u1,x),multiply(u2,p));
    return make_double2(__fma_rn(e,proposed.x,-hx.x),__fma_rn(e,proposed.y,-hx.y));
}
__global__ void fill_guard(double2 *g,size_t n){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n)g[i]=make_double2(0.75e20,-0.125e20);
}
__global__ void six_ordered(Grid q,const double2 *x,const double2 *p,const double2 *hx,
                            const double2 *hp,double h,double2 *out){
    int v=blockIdx.x*blockDim.x+threadIdx.x;
    if(v>=q.nv)return;
    double2 a[6]={};
    for(int iz=0;iz<q.nz;++iz)for(int iy=0;iy<q.ny;++iy)for(int ix=0;ix<q.nx;++ix){
        size_t i=q.at(ix,iy,iz,v);
        double2 xx=x[i],pp=p[i],hhx=hx[i],hhp=hp[i];
        a[0]=sum(a[0],conjugate_product(xx,xx));
        a[1]=sum(a[1],conjugate_product(pp,xx));
        a[2]=sum(a[2],conjugate_product(pp,pp));
        a[3]=sum(a[3],conjugate_product(xx,hhx));
        a[4]=sum(a[4],conjugate_product(pp,hhx));
        a[5]=sum(a[5],conjugate_product(pp,hhp));
    }
    for(int k=0;k<6;++k)out[6*v+k]=make_double2(__dmul_rn(a[k].x,h),__dmul_rn(a[k].y,h));
}
__global__ void gradient_points(Grid q,const double2 *x,const double2 *p,double2 *hx,
        const double2 *hp,const double2 *u,const double *e,double2 *g){
    size_t id=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t physical=size_t(q.nx)*q.ny*q.nz;
    if(id>=physical*q.nv)return;
    int v=id/physical;size_t j=id%physical;
    int ix=j%q.nx,iy=(j/q.nx)%q.ny,iz=j/(size_t(q.nx)*q.ny);
    size_t i=q.at(ix,iy,iz,v);
    double2 hhx=hx[i];
    g[i]=updated_gradient(x[i],p[i],hhx,hp[i],u[2*v],u[2*v+1],e[v]);
    hx[i]=hhx;
}
template<bool Fused>
__global__ void gradient_norm(Grid q,const double2 *x,const double2 *p,double2 *hx,
        const double2 *hp,const double2 *u,const double *e,double2 *g,double h,double2 *out){
    int v=blockIdx.x*blockDim.x+threadIdx.x;
    if(v>=q.nv)return;
    double2 a=make_double2(0.,0.);
    for(int iz=0;iz<q.nz;++iz)for(int iy=0;iy<q.ny;++iy)for(int ix=0;ix<q.nx;++ix){
        size_t i=q.at(ix,iy,iz,v);
        double2 gg;
        if constexpr(Fused){
            double2 hhx=hx[i];
            gg=updated_gradient(x[i],p[i],hhx,hp[i],u[2*v],u[2*v+1],e[v]);
            hx[i]=hhx;g[i]=gg;
        } else gg=g[i];
        a=sum(a,conjugate_product(gg,gg));
    }
    out[v]=make_double2(__dmul_rn(a.x,h),__dmul_rn(a.y,h));
}


// Same complex multiply/add tree as the CPU update of x in gscg_zwf (where bk < 0).
// As on the CPU, an orbital whose step is rejected keeps x, but Hx is already updated.
__global__ void accepted_x(Grid q,double2 *x,const double2 *p,const double2 *u,const double *bk){
    size_t id=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t physical=size_t(q.nx)*q.ny*q.nz;
    if(id>=physical*q.nv)return;
    int v=id/physical;size_t j=id%physical;
    if(!(bk[v]<0.))return;
    int ix=j%q.nx,iy=(j/q.nx)%q.ny,iz=j/(size_t(q.nx)*q.ny);
    size_t i=q.at(ix,iy,iz,v);
    x[i]=sum(multiply(u[2*v],x[i]),multiply(u[2*v+1],p[i]));
}

struct Context {
    Grid q;
    size_t bytes,physical;
    double h;
    double2 *x=nullptr,*p=nullptr,*hx=nullptr,*hp=nullptr,*g=nullptr;
    double2 *u=nullptr,*six=nullptr,*norm=nullptr;
    double *e=nullptr;
    bool epoch=false,initial_hpsi=false,initial_done=false;
    int directions=0;
    EpochInput maps{};
    double2 *coeff=nullptr;
    size_t map_bytes=0,coeff_count=0;
};
extern "C" int salmon_cg_device_guard(){
    const char *visible=std::getenv("CUDA_VISIBLE_DEVICES");
    if(!visible||!*visible||std::strchr(visible,','))return -10;
    int count=0; auto rc=cudaGetDeviceCount(&count);
    if(rc!=cudaSuccess)return int(rc);
    return count==1?0:-11;
}
extern "C" int salmon_cg_close(void *handle){
    auto *c=static_cast<Context*>(handle);if(!c)return 0;
    int result=0;
    auto release=[&](const void *p){if(p){int rc=int(cudaFree(const_cast<void*>(p)));if(!result)result=rc;}};
    release(c->x);release(c->p);release(c->hx);release(c->hp);release(c->g);
    release(c->u);release(c->six);release(c->norm);release(c->e);
    const auto &m=c->maps;
    const void *owned[]={m.idx,m.idy,m.idz,m.vlocal,m.lap0,m.lapt,m.nabt,
                         m.ia_tbl,m.mps,m.jxyz,m.uv,m.rinv,m.v2j,m.v2nlma,m.k2ilma,m.k2j,c->coeff};
    for(const void *p:owned)release(p);
    delete c;return result;
}
extern "C" int salmon_cg_begin(int nx,int ny,int nz,int nv,int pady,double h,
 const double2 *x,const double2 *p,const double2 *hx,const double2 *hp,
 double2 *six,void **handle){
    if(!handle)return -1;*handle=nullptr;
    if(pady<0||pady>1||nx<=0||ny<=0||nz<=0||nv<=0||nv>std::numeric_limits<int>::max()-127||!x||!p||!hx||!hp||!six)return -1;
    int rc=salmon_cg_device_guard();if(rc)return rc;
    size_t cells=nx;
    for(size_t n:{size_t(ny)+pady,size_t(nz),size_t(nv)}){
        if(cells>std::numeric_limits<size_t>::max()/n)return -2;cells*=n;
    }
    if(cells>std::numeric_limits<size_t>::max()/sizeof(double2)||
       cells>size_t(std::numeric_limits<int>::max())*128)return -2;
    auto *c=new(std::nothrow) Context;if(!c)return -3;
    c->q=Grid{nx,ny,nz,0,nv,size_t(nx),size_t(ny)+pady,
              size_t(nx)*(size_t(ny)+pady),size_t(nx)*(size_t(ny)+pady)*nz};
    c->bytes=cells*sizeof(double2);c->physical=size_t(nx)*ny*nz*nv;c->h=h;
#define TRY(cmd) do{rc=int(cmd);if(rc){salmon_cg_close(c);return rc;}}while(0)
    for(double2 **a:{&c->x,&c->p,&c->hx,&c->hp,&c->g})TRY(cudaMalloc(reinterpret_cast<void**>(a),c->bytes));
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->u),size_t(nv)*2*sizeof(double2)));
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->six),size_t(nv)*6*sizeof(double2)));
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->norm),size_t(nv)*sizeof(double2)));
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->e),size_t(nv)*sizeof(double)));
    TRY(cudaMemcpy(c->x,x,c->bytes,cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(c->p,p,c->bytes,cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(c->hx,hx,c->bytes,cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(c->hp,hp,c->bytes,cudaMemcpyHostToDevice));
    six_ordered<<<(size_t(nv)+127)/128,128>>>(c->q,c->x,c->p,c->hx,c->hp,h,c->six);
    TRY(cudaGetLastError());
    TRY(cudaMemcpy(six,c->six,size_t(nv)*6*sizeof(double2),cudaMemcpyDeviceToHost));
    *handle=c;return 0;
#undef TRY
}
static int finish(void *handle,const double2 *u,const double *e,
 double2 *hx,double2 *g,double2 *norm,int fused,bool copy_hx){
    auto *c=static_cast<Context*>(handle);
    if(!c||!u||!e||!hx||!g||!norm||(fused!=0&&fused!=1))return -1;
    int rc;
#define TRY(cmd) do{rc=int(cmd);if(rc)return rc;}while(0)
    const size_t nv=c->q.nv;
    TRY(cudaMemcpy(c->u,u,2*nv*sizeof(double2),cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(c->e,e,nv*sizeof(double),cudaMemcpyHostToDevice));
    if(fused)gradient_norm<true><<<(nv+127)/128,128>>>(c->q,c->x,c->p,c->hx,c->hp,c->u,c->e,c->g,c->h,c->norm);
    else {
        gradient_points<<<(c->physical+127)/128,128>>>(c->q,c->x,c->p,c->hx,c->hp,c->u,c->e,c->g);
        TRY(cudaGetLastError());
        gradient_norm<false><<<(nv+127)/128,128>>>(c->q,c->x,c->p,c->hx,c->hp,c->u,c->e,c->g,c->h,c->norm);
    }
    TRY(cudaGetLastError());
    if(copy_hx)TRY(cudaMemcpy(hx,c->hx,c->bytes,cudaMemcpyDeviceToHost));
    // Copy only physical rows; leave caller gradient padding byte-for-byte untouched.
    TRY(cudaMemcpy2D(g,c->q.plane*sizeof(double2),c->g,c->q.plane*sizeof(double2),
                     size_t(c->q.nx)*c->q.ny*sizeof(double2),size_t(c->q.nz)*nv,cudaMemcpyDeviceToHost));
    TRY(cudaMemcpy(norm,c->norm,nv*sizeof(double2),cudaMemcpyDeviceToHost));
    return 0;
#undef TRY
}

// Existing one-step ABI and arithmetic stay available without residency.
extern "C" int salmon_cg_finish(void *handle,const double2 *u,const double *e,
 double2 *hx,double2 *g,double2 *norm,int fused){
    return finish(handle,u,e,hx,g,norm,fused,true);
}
extern "C" int salmon_cg_retained_finish(void *handle,const double2 *u,const double *e,
 double2 *hx,double2 *g,double2 *norm,int copy_hx){
    if(copy_hx!=0&&copy_hx!=1)return -1;
    return finish(handle,u,e,hx,g,norm,0,copy_hx!=0);
}
extern "C" int salmon_cg_refresh(void *handle,const double2 *p,const double2 *hp,double2 *six){
    auto *c=static_cast<Context*>(handle);if(!c||!p||!hp||!six)return -1;
    int rc=salmon_cg_device_guard();if(rc)return rc;
#define TRY(cmd) do{rc=int(cmd);if(rc)return rc;}while(0)
    // CPU direction/Hpsi remain the producers; x and Hx are already current here.
    TRY(cudaMemcpy(c->p,p,c->bytes,cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(c->hp,hp,c->bytes,cudaMemcpyHostToDevice));
    six_ordered<<<(size_t(c->q.nv)+127)/128,128>>>(c->q,c->x,c->p,c->hx,c->hp,c->h,c->six);
    TRY(cudaGetLastError());
    TRY(cudaMemcpy(six,c->six,size_t(c->q.nv)*6*sizeof(double2),cudaMemcpyDeviceToHost));
    return 0;
#undef TRY
}
extern "C" int salmon_cg_accept(void *handle,const double *bk){
    auto *c=static_cast<Context*>(handle);if(!c||!bk)return -1;
    int rc;
#define TRY(cmd) do{rc=int(cmd);if(rc)return rc;}while(0)
    // E has no device consumer after finish. Reuse that small buffer for bk.
    TRY(cudaMemcpy(c->e,bk,size_t(c->q.nv)*sizeof(double),cudaMemcpyHostToDevice));
    accepted_x<<<(c->physical+127)/128,128>>>(c->q,c->x,c->p,c->u,c->e);
    TRY(cudaGetLastError());
    TRY(cudaDeviceSynchronize());
    return 0;
#undef TRY
}
extern "C" int salmon_cg_snapshot(void *handle,double2 *x,double2 *hx){
    auto *c=static_cast<Context*>(handle);if(!c||!x||!hx)return -1;
    int rc=int(cudaMemcpy(x,c->x,c->bytes,cudaMemcpyDeviceToHost));if(rc)return rc;
    return int(cudaMemcpy(hx,c->hx,c->bytes,cudaMemcpyDeviceToHost));
}


// Whole-CG mode: x, Hx, gradient, direction and H(direction) stay on the GPU for a whole
// gscg_zwf call. Needs SALMON_CG_ORDERED_NONLOCAL at build time and SALMON_CG_CUDA_EPOCH=1.
__global__ void pair_ordered(Grid q,const double2 *a,const double2 *b,double h,double2 *out){
    int v=blockIdx.x*blockDim.x+threadIdx.x;if(v>=q.nv)return;
    double2 acc=make_double2(0.,0.);
    for(int iz=0;iz<q.nz;++iz)for(int iy=0;iy<q.ny;++iy)for(int ix=0;ix<q.nx;++ix){
        size_t i=q.at(ix,iy,iz,v);acc=sum(acc,conjugate_product(a[i],b[i]));
    }
    out[v]=make_double2(__dmul_rn(acc.x,h),__dmul_rn(acc.y,h));
}
__global__ void initial_gradient(Grid q,const double2 *x,const double2 *hx,const double2 *energy,double2 *g){
    size_t id=size_t(blockIdx.x)*blockDim.x+threadIdx.x,physical=size_t(q.nx)*q.ny*q.nz;
    if(id>=physical*q.nv)return;
    int v=id/physical;size_t j=id%physical;
    size_t i=q.at(j%q.nx,(j/q.nx)%q.ny,j/(size_t(q.nx)*q.ny),v);
    g[i]=make_double2(__fma_rn(energy[v].x,x[i].x,-hx[i].x),__fma_rn(energy[v].x,x[i].y,-hx[i].y));
}
__global__ void direction_points(Grid q,const double2 *g,double2 *p,const double *bk,bool first){
    size_t id=size_t(blockIdx.x)*blockDim.x+threadIdx.x,physical=size_t(q.nx)*q.ny*q.nz;
    if(id>=physical*q.nv)return;
    int v=id/physical;size_t j=id%physical;
    size_t i=q.at(j%q.nx,(j/q.nx)%q.ny,j/(size_t(q.nx)*q.ny),v);
    if(first)p[i]=g[i];
    else p[i]=make_double2(__fma_rn(bk[v],p[i].x,g[i].x),__fma_rn(bk[v],p[i].y,g[i].y));
}
extern "C" int salmon_cg_epoch_available(){
#ifdef SALMON_CG_ORDERED_NONLOCAL
    return 1;
#else
    return 0;
#endif
}
// Check that the inverse projector map of init_uvpsi_blocking lists every (ilma,j) visit in
// the original order. Runs once, when the context is created, not per Hpsi. Invalid maps are
// an error; unsupported layouts have already been sent to the CPU path by the bridge.
static bool valid_epoch_maps(const EpochInput &m){
    if(m.no<=0||m.nk<=0||m.nlma<=0||m.nps<=0||m.natom<=0||m.max_vi<=0||
       m.max_vi>4096||m.max_contrib<=0)return false;
    const void *required[]={m.idx,m.idy,m.idz,m.vlocal,m.lap0,m.lapt,m.nabt,
        m.ia_tbl,m.mps,m.jxyz,m.uv,m.rinv,m.v2j,m.v2nlma,m.k2ilma,m.k2j};
    for(const void *p:required)if(!p)return false;
    const int *index[]={m.idx,m.idy,m.idz};
    for(int axis=0;axis<3;++axis){
        if(m.hi[axis]-m.lo[axis]!=15)return false;
        for(int j=0;j<24;++j)if(index[axis][j]<m.lo[axis]||index[axis][j]>m.hi[axis])return false;
    }
    auto point=[&](const int *xyz){
        for(int axis=0;axis<3;++axis)if(xyz[axis]<m.lo[axis]||xyz[axis]>m.hi[axis])return -1;
        return xyz[0]-m.lo[0]+16*(xyz[1]-m.lo[1]+16*(xyz[2]-m.lo[2]));
    };
    std::vector<int> inverse(4096,-1),seen(size_t(m.max_vi),0);
    for(int vi=0;vi<m.max_vi;++vi){
        int at=point(m.v2j+3*vi);
        if(at<0||inverse[at]!=-1||m.v2nlma[vi]<1||m.v2nlma[vi]>m.max_contrib)return false;
        inverse[at]=vi;
    }
    for(int ia=0;ia<m.natom;++ia)if(m.mps[ia]<0||m.mps[ia]>m.nps)return false;
    for(int ilma=0;ilma<m.nlma;++ilma){
        int ia=m.ia_tbl[ilma]-1;if(ia<0||ia>=m.natom)return false;
        for(int j=0;j<m.mps[ia];++j){
            int at=point(m.jxyz+3*(size_t(j)+size_t(m.nps)*ia));if(at<0)return false;
            int vi=inverse[at];if(vi<0||seen[vi]>=m.v2nlma[vi])return false;
            size_t row=vi+size_t(m.max_vi)*seen[vi]++;
            if(m.k2ilma[row]!=ilma+1||m.k2j[row]!=j+1)return false;
        }
    }
    for(int vi=0;vi<m.max_vi;++vi)if(seen[vi]!=m.v2nlma[vi])return false;
    return true;
}
extern "C" int salmon_cg_epoch_begin(const EpochInput *input,double h,const double2 *x,void **handle){
    if(!handle)return -1;*handle=nullptr;
    if(!salmon_cg_epoch_available())return -40;
    if(!input||!x)return -41;
    try {if(!valid_epoch_maps(*input))return -41;}
    catch(const std::bad_alloc &){return -3;}
    const EpochInput &m=*input;
    size_t nv=size_t(m.no)*m.nk;
    if(nv>size_t(std::numeric_limits<int>::max())/32)return -2;
    const size_t limit=std::numeric_limits<size_t>::max()/sizeof(double2);
    if(nv>limit/(16*17*16)||nv>limit/size_t(m.nlma)||
       size_t(m.nps)>limit/size_t(m.nlma)/size_t(m.nk)||
       size_t(m.nps)>limit/size_t(m.natom)/3||
       size_t(m.max_vi)>limit/size_t(m.max_contrib))return -2;
    int rc=salmon_cg_device_guard();if(rc)return rc;
    auto *c=new(std::nothrow) Context;if(!c)return -3;
    c->q=Grid{16,16,16,0,int(nv),16,17,16*17,16*17*16};
    c->bytes=nv*16*17*16*sizeof(double2);c->physical=nv*4096;c->h=h;c->epoch=true;
    // Copy only scalar metadata first: borrowed host pointers must never be freed.
    for(int axis=0;axis<3;++axis){c->maps.lo[axis]=m.lo[axis];c->maps.hi[axis]=m.hi[axis];}
    c->maps.no=m.no;c->maps.nk=m.nk;c->maps.nlma=m.nlma;c->maps.nps=m.nps;
    c->maps.natom=m.natom;c->maps.max_vi=m.max_vi;c->maps.max_contrib=m.max_contrib;
#define TRY(cmd) do{rc=int(cmd);if(rc){salmon_cg_close(c);return rc;}}while(0)
    for(double2 **a:{&c->x,&c->p,&c->hx,&c->hp,&c->g})TRY(cudaMalloc(reinterpret_cast<void**>(a),c->bytes));
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->u),nv*2*sizeof(double2)));
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->six),nv*6*sizeof(double2)));
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->norm),nv*sizeof(double2)));
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->e),nv*sizeof(double)));
    c->coeff_count=nv*size_t(m.nlma);
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->coeff),c->coeff_count*sizeof(double2)));
    auto copy=[&](auto &dst,const auto *src,size_t n){
        using T=std::remove_const_t<std::remove_pointer_t<std::decay_t<decltype(dst)>>>;
        T *device=nullptr;int result=int(cudaMalloc(reinterpret_cast<void**>(&device),n*sizeof(T)));
        if(result)return result;dst=device;c->map_bytes+=n*sizeof(T);
        return int(cudaMemcpy(device,src,n*sizeof(T),cudaMemcpyHostToDevice));
    };
    TRY(copy(c->maps.idx,m.idx,24));TRY(copy(c->maps.idy,m.idy,24));TRY(copy(c->maps.idz,m.idz,24));
    TRY(copy(c->maps.vlocal,m.vlocal,4096));TRY(copy(c->maps.lap0,m.lap0,m.nk));
    TRY(copy(c->maps.lapt,m.lapt,12));TRY(copy(c->maps.nabt,m.nabt,size_t(12)*m.nk));
    TRY(copy(c->maps.ia_tbl,m.ia_tbl,m.nlma));TRY(copy(c->maps.mps,m.mps,m.natom));
    TRY(copy(c->maps.jxyz,m.jxyz,size_t(3)*m.nps*m.natom));
    TRY(copy(c->maps.uv,m.uv,size_t(m.nps)*m.nlma*m.nk));TRY(copy(c->maps.rinv,m.rinv,m.nlma));
    TRY(copy(c->maps.v2j,m.v2j,size_t(3)*m.max_vi));TRY(copy(c->maps.v2nlma,m.v2nlma,m.max_vi));
    TRY(copy(c->maps.k2ilma,m.k2ilma,size_t(m.max_vi)*m.max_contrib));
    TRY(copy(c->maps.k2j,m.k2j,size_t(m.max_vi)*m.max_contrib));
    TRY(cudaMemcpy(c->x,x,c->bytes,cudaMemcpyHostToDevice));
    *handle=c;return 0;
#undef TRY
}
extern "C" int salmon_cg_epoch_query(void *handle,size_t *field_bytes,size_t *map_bytes,size_t *coeff_bytes){
    auto *c=static_cast<Context*>(handle);if(!c||!field_bytes||!map_bytes||!coeff_bytes)return -1;
    *field_bytes=5*c->bytes;*map_bytes=c->map_bytes;*coeff_bytes=c->coeff_count*sizeof(double2);
    return c->epoch?1:0;
}
extern "C" int salmon_cg_epoch_hpsi(void *handle,int initial){
    auto *c=static_cast<Context*>(handle);if(!c||!c->epoch||(initial!=0&&initial!=1))return -1;
    if(initial?(c->initial_hpsi||c->directions!=0):(!c->initial_done||c->directions==0))return -42;
#ifdef SALMON_CG_ORDERED_NONLOCAL
    const auto &m=c->maps;const double2 *psi=initial?c->x:c->p;double2 *hpsi=initial?c->hx:c->hp;
    int rc=salmon_cg_stencil_ordered(m.lo[0],m.hi[0],m.lo[1],m.hi[1],m.lo[2],m.hi[2],
        m.lo[0],m.hi[0],m.lo[1],m.hi[1]+1,m.lo[2],m.hi[2],1,m.no,m.nk,
        m.idx,m.idy,m.idz,m.vlocal,m.lap0,m.lapt,m.nabt,psi,hpsi);
    if(rc)return rc;
    rc=salmon_cg_nonlocal_ordered(hpsi,1,1,1,m.nk,1,m.no,1,m.nlma,m.nps,m.natom,
        m.lo[0],m.hi[0],m.lo[1],m.hi[1]+1,m.lo[2],m.hi[2],m.ia_tbl,m.mps,m.jxyz,
        m.uv,m.rinv,psi,m.max_vi,m.v2j,m.v2nlma,m.k2ilma,m.k2j,c->coeff,c->coeff_count);
    if(!rc&&initial)c->initial_hpsi=true;
    return rc;
#else
    return -40;
#endif
}
extern "C" int salmon_cg_epoch_initial(void *handle,double2 *energy,double2 *norm){
    auto *c=static_cast<Context*>(handle);
    if(!c||!c->epoch||!c->initial_hpsi||c->initial_done||!energy||!norm)return -42;
    int rc;size_t nv=c->q.nv;
#define TRY(cmd) do{rc=int(cmd);if(rc)return rc;}while(0)
    pair_ordered<<<(nv+127)/128,128>>>(c->q,c->x,c->hx,c->h,c->norm);TRY(cudaGetLastError());
    // Return the full complex <x|Hx>; the CPU takes its real part as before.
    TRY(cudaMemcpy(energy,c->norm,nv*sizeof(double2),cudaMemcpyDeviceToHost));
    initial_gradient<<<(c->physical+127)/128,128>>>(c->q,c->x,c->hx,c->norm,c->g);TRY(cudaGetLastError());
    pair_ordered<<<(nv+127)/128,128>>>(c->q,c->g,c->g,c->h,c->norm);TRY(cudaGetLastError());
    TRY(cudaMemcpy(norm,c->norm,nv*sizeof(double2),cudaMemcpyDeviceToHost));
    c->initial_done=true;return 0;
#undef TRY
}
extern "C" int salmon_cg_epoch_norm(void *handle,double2 *norm){
    auto *c=static_cast<Context*>(handle);if(!c||!c->epoch||!c->initial_done||!norm)return -42;
    pair_ordered<<<(size_t(c->q.nv)+127)/128,128>>>(c->q,c->g,c->g,c->h,c->norm);
    int rc=int(cudaGetLastError());if(rc)return rc;
    return int(cudaMemcpy(norm,c->norm,size_t(c->q.nv)*sizeof(double2),cudaMemcpyDeviceToHost));
}
extern "C" int salmon_cg_epoch_direction(void *handle,int iteration,const double *bk){
    auto *c=static_cast<Context*>(handle);
    if(!c||!c->epoch||!c->initial_done||iteration!=c->directions+1||!bk)return -42;
    int rc;
    if(iteration>1){rc=int(cudaMemcpy(c->e,bk,size_t(c->q.nv)*sizeof(double),cudaMemcpyHostToDevice));if(rc)return rc;}
    direction_points<<<(c->physical+127)/128,128>>>(c->q,c->g,c->p,c->e,iteration==1);
    rc=int(cudaGetLastError());if(rc)return rc;
    rc=int(cudaDeviceSynchronize());if(!rc)++c->directions;return rc;
}
extern "C" int salmon_cg_epoch_products(void *handle,double2 *six){
    auto *c=static_cast<Context*>(handle);if(!c||!c->epoch||c->directions==0||!six)return -42;
    six_ordered<<<(size_t(c->q.nv)+127)/128,128>>>(c->q,c->x,c->p,c->hx,c->hp,c->h,c->six);
    int rc=int(cudaGetLastError());if(rc)return rc;
    return int(cudaMemcpy(six,c->six,size_t(c->q.nv)*6*sizeof(double2),cudaMemcpyDeviceToHost));
}
extern "C" int salmon_cg_epoch_finish(void *handle,const double2 *u,const double *e,double2 *norm){
    auto *c=static_cast<Context*>(handle);if(!c||!c->epoch||!u||!e||!norm)return -1;
    int rc;size_t nv=c->q.nv;
#define TRY(cmd) do{rc=int(cmd);if(rc)return rc;}while(0)
    TRY(cudaMemcpy(c->u,u,2*nv*sizeof(double2),cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(c->e,e,nv*sizeof(double),cudaMemcpyHostToDevice));
    gradient_points<<<(c->physical+127)/128,128>>>(c->q,c->x,c->p,c->hx,c->hp,c->u,c->e,c->g);
    TRY(cudaGetLastError());
    gradient_norm<false><<<(nv+127)/128,128>>>(c->q,c->x,c->p,c->hx,c->hp,c->u,c->e,c->g,c->h,c->norm);
    TRY(cudaGetLastError());
    TRY(cudaMemcpy(norm,c->norm,nv*sizeof(double2),cudaMemcpyDeviceToHost));return 0;
#undef TRY
}
// x was initialized including padding; the other fields only own physical rows.
// Caller owns padding and it is never read from uninitialized device storage.
extern "C" int salmon_cg_epoch_read(void *handle,int field,double2 *out){
    auto *c=static_cast<Context*>(handle);if(!c||!c->epoch||!out||field<0||field>4)return -1;
    if(field>=3&&c->directions==0)return -42;
    const double2 *src[]={c->x,c->hx,c->g,c->p,c->hp};
    if(field==0)return int(cudaMemcpy(out,src[field],c->bytes,cudaMemcpyDeviceToHost));
    return int(cudaMemcpy2D(out,c->q.plane*sizeof(double2),src[field],c->q.plane*sizeof(double2),
        size_t(c->q.nx)*c->q.ny*sizeof(double2),size_t(c->q.nz)*c->q.nv,cudaMemcpyDeviceToHost));
}
