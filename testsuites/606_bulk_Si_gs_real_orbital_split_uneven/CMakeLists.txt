create_mpi_test()
# Compared with the layout reference 605, which CTest runs and verifies first.
# create_mpi_test() adds no tests to a build without MPI.
if (USE_MPI)
  set_tests_properties(run_606_bulk_Si_gs_real_orbital_split_uneven
                       PROPERTIES FIXTURES_SETUP run_606_bulk_Si_gs_real_orbital_split_uneven)
  set_tests_properties(prep_606_bulk_Si_gs_real_orbital_split_uneven
                       PROPERTIES FIXTURES_REQUIRED real_orbital_layout_reference)
endif ()
