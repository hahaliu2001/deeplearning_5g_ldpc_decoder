# deeplearning_5g_ldpc_decoder

## how to debug C code from python script
- Run this command in your terminal to allow GDB to work without root privileges for the current session:
```
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
```
- run the python code to before call C funciton
- check python process id
    - {os.getpid()
- run "(gdb) python Attach" and then select prrcessid 
