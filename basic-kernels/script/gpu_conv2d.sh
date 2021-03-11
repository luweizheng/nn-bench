#!/bin/bash

source activate torch16

cd ../pytorch

metrics="sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active,\
smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,\
smsp__cycles_elapsed.sum,\
smsp__cycles_elapsed.sum.per_second,\
smsp__pipe_tensor_op_hmma_cycles_active.sum,\
smsp__pipe_tensor_op_hmma_cycles_active.sum.per_second,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum,\
l1tex__t_set_accesses_pipe_tex_mem_surface_op_atom.sum,\
l1tex__t_set_accesses_pipe_tex_mem_surface_op_red.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,\
smsp__inst_executed_op_shared_ld.sum,\
smsp__inst_executed_op_shared_st.sum,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
lts__t_sectors_op_atom.sum,\
lts__t_sectors_op_red.sum,\
dram__sectors_read.sum,\
dram__sectors_write.sum
"

nsight="/usr/local/cuda/nsight-compute-2019.4.0/"
batch_size=64
name="conv2d"
prec="float16"


for metric in ${metrics}; do
    profilestring=$nsight"nv-nsight-cu-cli --profile-from-start off --metrics ${metric} -f"
    profilestring=${profilestring}" -o profile.name_${name}.batchsize_${batch_size}.inputshape_${input_shape}.kernelshape_${kernel_shape}.stride_${stride}.dataformat_.fp${prec}.pass_${ctype}"

    ${profilestring} $(which python) -u ./conv2d.py \
                --platform gpu
                --dtype float16 
done