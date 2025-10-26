# COREFL: A compressible reactive flow solver on generalized curvilinear coordinates

COREFL performs direct numerical simulations of compressible reactive flows on GPU based on finite difference method.

## Environment requirements
- **CUDA compiler**: supporting C++17 (nvcc > 11.0)
- **C++ compiler**: supporting C++20
- **MPI library**: supporting CUDA-aware MPI (E.g., OpenMPI > 1.8)
- **CMake**: supporting CUDA language
- **fmtlib** of C++

comments on the environments:

1. MPI: Only a few vendors' MPI support CUDA-aware MPI, and only on Linux systems. Therefore, only Linux system supports the parallel running of COREFL. However, any MPI version supports the compilation and running in serial modes.
2. fmt: This is a lib for outputting information from CPU, not GPU. In future releases, this would be get rid of. For now, you may need to compile the fmt lib by yourself according to the following instructions.

COREFL has been compiled and tested on Nvidia P100, V100, A100 GPUs.
The most frequently used configuration by us on A100 is given for reference: 

**CUDA 11.8 / gcc 11.3 / openmpi 4.1.5 / cmake 3.26.3**

## Compilation

The structure of the current folder is:

- depends/
  - include/
    - fmt/*
  - lib/
    - debug/*
    - release/*
- src/
  - gxl_lib/*
  - stat_lib/*
  - *
- CMakeLists.txt

### The compilation of *fmtlib*

For various environments, we need the fmtlib to be compiled first.

1. Download the fmt zip file from https://github.com/fmtlib/fmt/releases. For example, https://github.com/fmtlib/fmt/releases/download/11.2.0/fmt-11.2.0.zip.
2. Put it on the target platform and unzip.
3. In the command line, navigate to the unzipped folder containing them. Load the corresponding environment variables for running COREFL. For example, in our system, we use `module load mpi/openmpi4.1.5-gcc11.3.0-cuda11.8-ucx1.12.1 cmake/3.26.3` to load the compilers.
4. `cmake -Bbuild -DCMAKE_BUILD_TYPE=Release` The `build` after -B is the build folder containing build info. And the CMAKE_BUILD_TYPE must be specified. Therefore, we strongly advise you to use exactly this instruction.
5. `cmake --build build --parallel 16` The `build` after `--build` is the build folder created above. `16` is the number of threads used to compile the code.
6. After the build, there will be a `libfmt.a` file in the build folder. Copy and replace the fmtlib.a in our folder, whose path is gibven by `depends/lib/release/libfmt.a`.
7. The included files should also be kept consistent. Copy and replace all files from `./include/fmt/*` to the folder in COREFL `<coreflFolder>/depends/include/fmt/*`.

Now we have finished compiling fmt, which should be used for running COREFL.

### The compilation of COREFL

1. Navigate to the COREFL folder.
2. Modify the CMakeLists.txt:
   1. Modify the number in `set(CMAKE_CUDA_ARCHITECTURES 60)` according to the GPU compute capability. For example, this number is 60 for P100, 70 for V100, and 80 for A100.
   2. Modify `add_compile_definitions(MAX_SPEC_NUMBER=9)` according to the problem. The number should be larger than or equal to the species number to be used in computations. If no species is included, set it to 1.
   3. Modify `add_compile_definitions(MAX_REAC_NUMBER=19)` according to the problem. The number should be larger than or equal to the reaction number to be used in computations. If no reaction is included, set it to 1.
3. Load the compilation environment. For example, `module load mpi/openmpi4.1.5-gcc11.3.0-cuda11.8-ucx1.12.1 cmake/3.26.3`
4. `cmake -Bbuild -DCMAKE_BUILD_TYPE=Release`
5. `cmake --build build --parallel 16`

The executable COREFL should appear in ths current folder.

### The use of readGrid

COREFL reads the structured multiblock grid files in `Plot3D` format. We can not partition the blocks automatically. Therefore, if you want a parallel computation with multiple blocks, you need to partition the blocks manually.

The grid files read by COREFL is not the `Plot3D` file outputted directly from softwares, but the ones treated by our **readGrid** tool. 

The **readGrid** tool will read the "`gridFile.dat`" and "`gridFile.inp`", where "`gridFile`" is the filename, and output two folders `grid` and `boundary_condition`, which should be moved to the `input` folder and read by COREFL.

The advantage of using such a file is that the corresponding processes need only read their own blocks instead of waiting for asigning. But this may be improved or integrated in COREFL in the future.

This part is not included in the current file folder because the only example I give below has been treated with this tool. If you want to generate your own grid and want to use COREFL, feel free to contact me at guoxinliang@buaa.edu.cn or guoxinliang_buaa@163.com.

## Running

A typical folder for running COREFL is as follows:
- input/
  - grid/*
  - boundary_condition/*
  - chemistry/*
  - setup.txt
- run.sh
- corefl(optional)

All settings are set in the `setup.txt` file in the following manner:

`type name = value`

There should be space between each two symbols.

As we always run COREFL on clusters, we need a script file to run it. The executable is optional because we can specify the path to the exectuble in the script.

In our environment, we have 4 Nvidia A100s on a node, and 4 nodes. We can write the script as follows:
```bash
#!/bin/bash
#SBATCH --job-name=case1    # name of the job
#SBATCH --nodes=2           # number of nodes to use
#SBATCH --ntasks-per-node=4 # number of tasks per nodes
#SBATCH --gres=gpu:4        # number of GPUs per nodes
#SBATCH --qos=gpugpu        # included when more than 1 node is used

module purge
module load mpi/openmpi4.1.5-gcc11.3.0-cuda11.8-ucx1.12.1

### Job ID
JOB_ID="${SLURM_JOB_ID}"
### hosfile
HOSTFILE="hostfile.${JOB_ID}"
GPUS=4

for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  echo "${host[$k]} slots=$GPUS" >> $HOSTFILE
done

mpirun -n 8 \ # total number of processes to be started
  --mca btl tcp,self \
  --mca btl_tcp_if_include eth0 \
  --mca pml ob1 \
  --mca btl_base_warn_component_unused 0 \
  --hostfile ${HOSTFILE} \
  /path/to/corefl
```

With the above script, the corefl will be started. An `output` folder will be created which contains all output files.

In the output folder, a file named `flowfield.plt` will be created, which is the instantaneous flowfield file. A folder called `message` will also be created. The flowfield file, and the message folder, are necessary for starting a computation with existing results.

## Example setup

### Reactive shock tube

The reactive shock tube is presented as an example case because the grid file is small and easy to upload.

The grid file and boundary condition files have already been treated with our **readGrid** tool, and the folder is as follows:

- case/1-reactiveShockTube/
  - input/
    - grid/grid   0.dat
    - boundary_condition/
      - boundary   0.txt
      - inner   0.txt
      - parallel   0.txt
    - chemistry/
      - therm.dat
      - tran.dat
      - H2PREMIX.inp
    - setup.txt
  - run.sh

With these files, and a compiled executable corefl, the test can be started.

### Settings

Maybe you care about the settings, and let me explain the ones related to the current simulation in brief.

First, about the controls:
```c++
int gridIsBinary = 1  // the grid file is in ASCII (0) or binary (1).
real gridScale = 1    // In which scale is the grid generated. If the grid is generated in millimeters, the value is 0.001. In this case, we generate it in meters (1).
int total_step = 100000   // Total steps to compute
int output_file = 10000   // Frequency of outputting flowfield files
int output_screen = 1000  // Frequency of outputting residual info on screen
int output_time_series = 1000 // Frequency of outputting a flowfield named by the physical time. Because we want to compare some transient info with exsiting profiles. If this value is 0, no time series will be outputted.
```

Next, about the temporal schemes:
```c++
bool steady = 0     // If the simulation is steady or not. Steady (1), transient (0)
real dt = 1e-9      // The physical time step in second.
real total_simulation_time = 2.4e-4 // We have data at 0.23ms, so we want the computation to stop after that
```

Third, about the spatial schemes:
```c++
int shock_sensor = 2   // Ducros sensor (0), modified Jameson sensor (1), sensor based on density and pressure jump (2)
real shockSensor_threshold = -0.2 // A negative value means all points are computed by WENO scheme.
int viscous_order = 2   // inviscid(0), 2nd order (2), 8th order (8)
```

Fourth, about the chemistry
```c++
int species = 1     // Air (0), multi-component simulation (1)
string mechanism_file = chemistry/H2PREMIX.inp  // The path is relative to the "input/" folder.
int reaction = 1    // No reaction (0), Finite rate chemistry based on the mechanism (1)
```

Fifth, about the boundary conditions.
```c++
array string boundary_conditions {
  wall  outflow   // Write all boundary conditions' names here
}
struct wall {
  string type = wall  // Specify the type of this bc
  int label = 2       // This label must be consistent with the label of the bc when generating grid
  string  thermal_type    =   adiabatic  // Thermal wall type can be 1. "adiabatic" wall; 2. "isothermal" wall
  real    temperature     =   300        // If the wall is isothermal, the temperature should be given. As the wall in this case is adiabatic, this value will not be used.
}
struct outflow {
  string type = outflow
  int label   =   6
}

// other info about the flow in this case
string  reference_state =   left  // Specify the reference state for the simulation.
string default_init = left  // The default initialization info for the whole flowfield
struct  left {
    string  type            =   inflow
    int     label           =   5
    int     inflow_type     =   0   // 0 for constant inflow, 1 for profile inflow
    real    density         =   0.072
    real    velocity        =   0
    real    pressure        =   7173
    real    u               =   1
    real    v               =   0
    real    w               =   0
    real    H2              =   0.012772428
    real    O2              =   0.101362139
    real    AR              =   0.885865433
}
int groups_init = 2 // Because this case needs two parts with different conditions, we use this initialize in group function of COREFL. The restriction is that the names of groups other than the "default_init" must be named as "init_cond_l", where "l" is indexed from 0 to "group_init-1".
struct init_cond_0 {
	real x0 = 0.06
	real x1 = 0.15
	real y0 = -1
	real y1 = 1
	real z0 = -2
	real z1 = 2
	string name = right  // This tells the code to find a struct named as this value
}
struct  right {
    string  type            =   inflow
    int     label           =   5
    int     inflow_type     =   0   // 0 for constant inflow, 1 for profile inflow
    real    density     =   0.18075
    real    velocity            =   487.34
    real    pressure        =   35594
    real    u               =   -1
    real    v               =   0
    real    w               =   0
    real    H2              =   0.012772428
    real    O2              =   0.101362139
    real    AR              =   0.885865433
}
```

## Other comments 

If anyone is interested in our code in more cases, you can email me at guoxinliang@buaa.edu.cn freely. I would be happy to cooperate with you for more usages.

Below are some nonsense of mine.

The exampled one is a very simple case, but illustrates many interesting functions such as initialization in group of COREFL. We did not include the readGrid codes in this repository currently because the folder of the above case does not use that.

I have to admit that, because I am a PhD student in the fifth year, a more detailed introduction may cost more energy. If you want to use them, let us talk in private. Maybe I can give you some existing cases to save your time.

Besides, the compilation and environment issues always occur. If there are troubles, please also contact me at the email. I really want my effort of 2-3 years to be used instead of protected in my own computer.

Anyway, we have already conducted many researches based on our GPU code, which is much faster than the CPU codes. It is really fascinating to run a DNS within several days instead of weaks or months.
