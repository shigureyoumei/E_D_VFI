#!/bin/bash
#------- qsub option -----------
#PBS -A HAIRDESC
#PBS -q gen_S
#PBS -b 8
#PBS -l elapstim_req=24:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=4.1.6/gcc11.4.0-cuda11.8.0
#PBS -v OMP_NUM_THREADS=8
#PBS -N original_train_3skips
#PBS -j o

#------- Program execution -----------

cd $PBS_O_WORKDIR

echo "========== Job info =========="
echo "PBS_JOBID       = $PBS_JOBID"
echo "PBS_JOBNAME     = $PBS_JOBNAME"
echo "PBS_O_WORKDIR   = $PBS_O_WORKDIR"
echo "PBS_NODEFILE    = $PBS_NODEFILE"
echo "NQSV_MPIOPTS    = $NQSV_MPIOPTS"
echo "Start time      = $(date)"
echo "=============================="

# 1. load modules
module purge
module load cuda/11.8.0
module load openmpi/${NQSV_MPI_VER}

# 2. activate conda environment
CONDA_BASE=/work/XRAYDIFF/naran/miniconda3
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate EDVFI1

echo "========== Environment check =========="
echo "CONDA_BASE   = ${CONDA_BASE}"
echo "CONDA_PREFIX = ${CONDA_PREFIX}"
echo "Python       = $(which python)"
echo "Torchrun     = $(which torchrun)"
python -V
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu_count:', torch.cuda.device_count())"
echo "======================================="

# 3. project path and log setup
PROJECT_DIR=/work/HAIRDESC/naran/E_D_VFI
cd ${PROJECT_DIR}

LOG_DIR=${PROJECT_DIR}/logs
mkdir -p ${LOG_DIR}

JOB_NAME=${PBS_JOBNAME:-edvfi_gopro_ddp}
LOG_TIMESTAMP=$(date +%m_%d_%H_%M)
LOG_FILE=${LOG_DIR}/${LOG_TIMESTAMP}_${JOB_NAME}_log.txt

# Redirect all stdout/stderr to both PBS output and log file
exec > >(tee -a ${LOG_FILE}) 2>&1

echo "========== Log setup =========="
echo "Project dir     = ${PROJECT_DIR}"
echo "Log dir         = ${LOG_DIR}"
echo "Log file        = ${LOG_FILE}"
echo "==============================="

# 4. important distributed variables
# Pegasus: usually 1 node = 1 GPU
NNODES=$(sort -u ${PBS_NODEFILE} | wc -l)
NPROC_PER_NODE=1
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
MASTER_ADDR=$(head -n 1 ${PBS_NODEFILE})
MASTER_PORT=4321
export MASTER_ADDR MASTER_PORT WORLD_SIZE OMP_NUM_THREADS

echo "========== Distributed config =========="
echo "PBS_NP          = ${PBS_NP}"
echo "NNODES          = ${NNODES}"
echo "NPROC_PER_NODE  = ${NPROC_PER_NODE}"
echo "WORLD_SIZE      = ${WORLD_SIZE}"
echo "MASTER_ADDR     = ${MASTER_ADDR}"
echo "MASTER_PORT     = ${MASTER_PORT}"
echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS}"
echo ""
echo "Allocated nodes:"
cat ${PBS_NODEFILE}
echo ""
echo "Unique nodes:"
sort -u ${PBS_NODEFILE}
echo "========================================"

if [ "${NNODES}" -gt 1 ]; then
  echo "[OK] Multi-node job detected: ${NNODES} nodes."
else
  echo "[WARN] Only one node detected. This is NOT multi-node training."
fi

# 5. option file
OPT=/work/HAIRDESC/naran/E_D_VFI/options/train/GoPro/Final_bidirectionEncoder_XXNet_1attenfusion_3skip.yml

echo "========== Training config =========="
echo "OPT             = ${OPT}"
echo "Python          = $(which python)"
echo "Torchrun        = $(which torchrun)"
echo "====================================="

# 6. run distributed training
# mpirun starts one torchrun process per node.
# torchrun then starts NPROC_PER_NODE process on each node.
mpirun ${NQSV_MPIOPTS} -np ${NNODES} -npernode 1 \
  -x PATH \
  -x LD_LIBRARY_PATH \
  -x CONDA_PREFIX \
  -x MASTER_ADDR \
  -x MASTER_PORT \
  -x WORLD_SIZE \
  -x OMP_NUM_THREADS \
  bash -c '
    NODE_RANK=${OMPI_COMM_WORLD_RANK}

    if [ "${NODE_RANK}" != "0" ]; then
      exec >/dev/null 2>&1
    fi

    echo "========== Node launch check =========="
    echo "Host                  = $(hostname)"
    echo "NODE_RANK             = ${NODE_RANK}"
    echo "OMPI_COMM_WORLD_RANK  = ${OMPI_COMM_WORLD_RANK}"
    echo "OMPI_COMM_WORLD_SIZE  = ${OMPI_COMM_WORLD_SIZE}"
    echo "CUDA_VISIBLE_DEVICES  = ${CUDA_VISIBLE_DEVICES}"
    echo "Python                = $(which python)"
    echo "Torchrun              = $(which torchrun)"
    echo "======================================="

    python - <<EOF
import os
import socket

print("========== Python / CUDA check ==========")
print("hostname:", socket.gethostname())
print("OMPI_COMM_WORLD_RANK:", os.environ.get("OMPI_COMM_WORLD_RANK"))
print("OMPI_COMM_WORLD_SIZE:", os.environ.get("OMPI_COMM_WORLD_SIZE"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

try:
    import torch
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("[ERROR] Torch/CUDA check failed:", repr(e))

print("=========================================")
EOF

    echo "[INFO] Starting torchrun on $(hostname), NODE_RANK=${NODE_RANK}"

    torchrun \
      --nnodes='"${NNODES}"' \
      --nproc_per_node='"${NPROC_PER_NODE}"' \
      --node_rank=${NODE_RANK} \
      --master_addr='"${MASTER_ADDR}"' \
      --master_port='"${MASTER_PORT}"' \
      basicsr/train.py \
      -opt '"${OPT}"' \
      --launcher pytorch

    EXIT_CODE=$?

    if [ ${EXIT_CODE} -eq 0 ]; then
      echo "[OK] torchrun finished successfully on $(hostname), NODE_RANK=${NODE_RANK}"
    else
      echo "[ERROR] torchrun failed on $(hostname), NODE_RANK=${NODE_RANK}, EXIT_CODE=${EXIT_CODE}"
    fi

    exit ${EXIT_CODE}
  '

TRAIN_EXIT_CODE=$?

echo "========== Final status =========="
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
  echo "[OK] Distributed training finished successfully."
else
  echo "[ERROR] Distributed training failed. EXIT_CODE=${TRAIN_EXIT_CODE}"
fi
echo "End time = $(date)"
echo "Log file = ${LOG_FILE}"
echo "=================================="

exit ${TRAIN_EXIT_CODE}
