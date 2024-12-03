# set -ex -o pipefail
run() {
    echo -e "\n\nRunning M$1 N$2 K$3"
    logfile="M$1_N$2_K$3.log"
    icpx -fsycl -Xs '-device pvc' joint_matrix_bf16_fill_k_cache.cpp -DMATRIX_M=$1 -DMATRIX_N=$2 -DMATRIX_K=$3 -DPREFETCH &&
        echo compiled! &&
        numactl -m0 -C0-51 time ./a.out 2>&1 | tee $logfile
    # icpx -fsycl joint_matrix_bf16_fill_k_cache.cpp -DMATRIX_M=$1 -DMATRIX_N=$2 -DMATRIX_K=$3 -DPREFETCH -DOOB && SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file" ./a.out 2>&1 | tee $logfile
}
rm -rf test.log
M_list=(1 4 1024 2048 4096)
for M in ${M_list[@]}; do
    run $M 4096 4096 | tee -a test.log
    # run $M 4096 16384 | tee -a test.log
    # run $M 16384 4096 | tee -a test.log
    # run $M 4096 4096 | tee -a test.log
    # run $M 4096 11008 | tee -a test.log
    # run $M 11008 4096 | tee -a test.log
    # run $M 5120 13824 | tee -a test.log
    # run $M 5120 13824 | tee -a test.log
    # run $M 13824 5120 | tee -a test.log
    # run $M 4096 6144 | tee -a test.log
    # run $M 4096 5504 | tee -a test.log
    # run $M 4096 5504 | tee -a test.log
    # run $M 5504 4096 | tee -a test.log
    # run $M 4096 128256 | tee -a test.log
    # run $M 2048 32000 | tee -a test.log
    # run $M 4096 128256 | tee -a test.log
done
