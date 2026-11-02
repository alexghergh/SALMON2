create_mpi_test()
# Compared with the layout reference 605, which CTest runs and verifies first.
# create_mpi_test() adds no tests to a build without MPI.
if (USE_MPI)
  set_tests_properties(run_608_bulk_Si_gs_real_rgrid_split
                       PROPERTIES FIXTURES_SETUP run_608_bulk_Si_gs_real_rgrid_split)
  set_tests_properties(prep_608_bulk_Si_gs_real_rgrid_split
                       PROPERTIES FIXTURES_REQUIRED real_orbital_layout_reference)
endif ()
