CUDA12_BIN ?= /usr/bin
PATH := $(CUDA12_BIN):$(PATH)
export PATH
NVCC ?= /usr/local/cuda/bin/nvcc

CUDAFLAGS = -O3 -std=c++14 -Xcompiler -fPIC -cudart shared -I ../c_lib
LDFLAGS = -shared -lm
EXE_LDFLAGS = -Xlinker -rpath -Xlinker '$$ORIGIN'
EXE_LIBS = -ldl

LIB_NAME = libnrLDPC_cuda.so
DECODER_EXE_NAME = test_ldpc_5g_decoder_cuda
ENCODER_EXE_NAME = test_ldpc_5g_encoder_cuda
CRC_EXE_NAME = test_nr_crc_cuda

SRCS = nr_crc_tables.cu \
       nr_crc.cu \
       ldpc_tables.cu \
       ldpc_internal.cu \
       ldpc_rate_recovery.cu \
       fixedpoint_ldpc_decoder.cu \
       degree_fixedpoint_ldpc_decoder.cu \
       c_ldpc_5g_decoder.cu \
       c_ldpc_5g_encoder.cu \
       ldpc_rate_match.cu \
       ldpc_encoder.cu

OBJS = $(SRCS:.cu=.o)

all: $(LIB_NAME) tests

tests: $(DECODER_EXE_NAME) $(ENCODER_EXE_NAME) $(CRC_EXE_NAME)

$(LIB_NAME): $(OBJS)
	$(NVCC) -o $@ $^ $(LDFLAGS)

$(DECODER_EXE_NAME): test_ldpc_5g_decoder.cu $(LIB_NAME)
	$(NVCC) $(CUDAFLAGS) -o $@ $< -L. -lnrLDPC_cuda $(EXE_LDFLAGS) $(EXE_LIBS)

$(ENCODER_EXE_NAME): test_ldpc_5g_encoder.cu $(LIB_NAME)
	$(NVCC) $(CUDAFLAGS) -o $@ $< -L. -lnrLDPC_cuda $(EXE_LDFLAGS) $(EXE_LIBS)

$(CRC_EXE_NAME): test_nr_crc.cu $(LIB_NAME)
	$(NVCC) $(CUDAFLAGS) -o $@ $< -L. -lnrLDPC_cuda $(EXE_LDFLAGS)

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

clean:
	rm -f *.o $(LIB_NAME) $(DECODER_EXE_NAME) $(ENCODER_EXE_NAME) $(CRC_EXE_NAME)

.PHONY: all tests clean
