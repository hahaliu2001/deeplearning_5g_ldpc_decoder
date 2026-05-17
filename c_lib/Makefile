CC = gcc
# Default flags
CFLAGS = -Wall -Wextra -fPIC
LDFLAGS = -shared -lm

LIB_NAME = libnrLDPC.so
EXE_NAME = test_ldpc_5g_decoder
ENCODE_EXE_NAME = test_ldpc_5g_encoder
SRCS = nr_crc.c \
	   ldpc_tables.c \
	   ldpc_internal.c \
	   ldpc_rate_recovery.c \
	   fixedpoint_ldpc_decoder.c \
	   degree_fixedpoint_ldpc_decoder.c \
	   BG1_trained_ldpc_weights.c \
	   BG2_trained_ldpc_weights.c \
	   neural_fixedpoint_ldpc_decoder.c \
	   floatpoint_ldpc_decoder.c \
	   c_ldpc_5g_decoder.c \
	   c_ldpc_5g_encoder.c \
	   tx_rx_process.c \
	   ldpc_rate_match.c \
	   ldpc_encoder.c \

# Default build (Release)
all: CFLAGS += -O3
all: $(LIB_NAME) $(EXE_NAME) $(ENCODE_EXE_NAME)
all: $(LIB_NAME) 

# Debug build
debug: CFLAGS += -g -O0 -DDEBUG
debug: $(LIB_NAME) $(EXE_NAME) $(ENCODE_EXE_NAME)

$(LIB_NAME): $(SRCS:.c=.o)
	$(CC)  -o $@ $^ $(LDFLAGS)


# $(EXE_NAME): test_ldpc_5g_decoder.c $(SRCS:.c=.o)
$(EXE_NAME): test_ldpc_5g_decoder.c $(LIB_NAME)
	$(CC) $(CFLAGS) -o $@ $^ 

$(ENCODE_EXE_NAME): test_ldpc_5g_encoder.c $(LIB_NAME)
	$(CC) $(CFLAGS) -o $@ $^ 

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(LIB_NAME) $(EXE_NAME) $(ENCODE_EXE_NAME)
	
.PHONY: all debug clean
