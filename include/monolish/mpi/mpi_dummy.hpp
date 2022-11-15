#pragma once
#ifndef OMPI_MPI_H
#define OMPI_MPI_H

#ifndef OMPI_DECLSPEC
#if defined(WIN32) || defined(_WIN32)
#if defined(OMPI_IMPORTS)
#define OMPI_DECLSPEC __declspec(dllimport)
#else
#define OMPI_DECLSPEC
#endif /* defined(OMPI_IMPORTS) */
#else
#if OPAL_C_HAVE_VISIBILITY == 1
#define OMPI_DECLSPEC __attribute__((visibility("default")))
#else
#define OMPI_DECLSPEC
#endif
#endif
#endif

OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_null;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_min;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_max;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_sum;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_prod;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_land;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_band;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_lor;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_bor;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_lxor;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_bxor;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_maxloc;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_minloc;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_replace;
OMPI_DECLSPEC extern struct ompi_predefined_op_t ompi_mpi_op_no_op;

using MPI_Status = int;
using MPI_Request = int;
// typedef struct ompi_status_public_t MPI_Status;
typedef struct ompi_communicator_t *MPI_Comm;
typedef struct ompi_datatype_t *MPI_Datatype;
typedef struct ompi_errhandler_t *MPI_Errhandler;
typedef struct ompi_file_t *MPI_File;
typedef struct ompi_group_t *MPI_Group;
typedef struct ompi_info_t *MPI_Info;
typedef struct ompi_op_t *MPI_Op;
typedef struct ompi_message_t *MPI_Message;
typedef struct ompi_win_t *MPI_Win;
typedef struct mca_base_var_enum_t *MPI_T_enum;
typedef struct ompi_mpit_cvar_handle_t *MPI_T_cvar_handle;
typedef struct mca_base_pvar_handle_t *MPI_T_pvar_handle;
typedef struct mca_base_pvar_session_t *MPI_T_pvar_session;

#if !OMPI_BUILDING
#if defined(c_plusplus) || defined(__cplusplus)
#define OMPI_PREDEFINED_GLOBAL(type, global)                                   \
  (static_cast<type>(static_cast<void *>(&(global))))
#else
#define OMPI_PREDEFINED_GLOBAL(type, global) ((type)((void *)&(global)))
#endif
#else
#define OMPI_PREDEFINED_GLOBAL(type, global) ((type) & (global))
#endif

#define MPI_MAX OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_max)
#define MPI_MIN OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_min)
#define MPI_SUM OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_sum)
#define MPI_PROD OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_prod)
#define MPI_LAND OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_land)
#define MPI_BAND OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_band)
#define MPI_LOR OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_lor)
#define MPI_BOR OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_bor)
#define MPI_LXOR OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_lxor)
#define MPI_BXOR OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_bxor)
#define MPI_MAXLOC OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_maxloc)
#define MPI_MINLOC OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_minloc)
#define MPI_REPLACE OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_replace)
#define MPI_NO_OP OMPI_PREDEFINED_GLOBAL(MPI_Op, ompi_mpi_op_no_op)

#endif /* OMPI_MPI_H */
