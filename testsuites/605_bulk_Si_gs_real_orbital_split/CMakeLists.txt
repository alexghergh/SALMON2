create_mpi_test()
# 605 is the layout reference that 606-608 compare with (fixture real_orbital_layout_reference).
# create_mpi_test() adds no tests to a build without MPI.
if (USE_MPI)
  set_tests_properties(run_605_bulk_Si_gs_real_orbital_split
                       PROPERTIES FIXTURES_SETUP run_605_bulk_Si_gs_real_orbital_split)
  set_tests_properties(verify_605_bulk_Si_gs_real_orbital_split
                       PROPERTIES FIXTURES_SETUP real_orbital_layout_reference)
endif ()
