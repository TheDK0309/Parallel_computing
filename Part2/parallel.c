/* COMP.CE.350 Parallelization Excercise 2021
   Copyright (c) 2016 Matias Koskela matias.koskela@tut.fi
                      Heikki Kultala heikki.kultala@tut.fi
                      Topi Leppanen  topi.leppanen@tuni.fi

VERSION 1.1 - updated to not have stuck satellites so easily
VERSION 1.2 - updated to not have stuck satellites hopefully at all.
VERSION 19.0 - make all satellites affect the color with weighted average.
               add physic correctness check.
VERSION 20.0 - relax physic correctness check
*/

// Example compilation on linux
// no optimization:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm
// most optimizations: gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2
// +vectorization +vectorize-infos: gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec
// +math relaxation:  gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math
// prev and OpenMP:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math -fopenmp
// prev and OpenCL:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math -fopenmp -lOpenCL

// Example compilation on macos X
// no optimization:   gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL
// most optimization: gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL -O3
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h> // printf
#include <math.h> // INFINITY
#include <stdlib.h>
#include <string.h>

#include <CL/opencl.h> 
#include "parallel.h" 

// Window handling includes
#ifndef __APPLE__
#include <GL/gl.h>
#include <GL/glut.h>
#else
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#endif

// Is used to find out frame times
int previousFrameTimeSinceStart = 0;
int previousFinishTime = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;

// Pixel buffer which is rendered to the screen
color* pixels;

// Pixel buffer which is used for error checking
color* correctPixels;

// Buffer for all satelites in the space
satellite* satellites;
satellite* backupSatelites;

// ## You may add your own variables here ##
cl_command_queue physics_queue = NULL;
cl_command_queue graphics_queue = NULL;

cl_context physics_context = NULL;
cl_context graphics_context = NULL;

cl_mem physics_satellites_buffer = NULL;
cl_mem graphics_satellites_buffer = NULL;
cl_mem pixels_buffer = NULL;

cl_program physics_program = NULL;
cl_program graphics_program = NULL;

cl_kernel physics_kernel = NULL;
cl_kernel graphics_kernel = NULL;

cl_int status;

size_t local[2];

void chk(cl_int status, const char* cmd) {

    if (status != CL_SUCCESS) {
        printf("%s failed (%d)\n", cmd, status);
        exit(-1);
    }
}

// Get the ID of the desired device type.
cl_device_id getDeviceID(cl_device_type device_type) {
    cl_platform_id* platforms;

    // Discovery platform
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);

    //allocate space enough for each platforms
    cl_platform_id platform_id = NULL;
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    chk(status, "clGetPlatformIDs");

    // Get the 1st device
    cl_uint numDevices = 0;
    for (int i = 0; i < numPlatforms; ++i) {
        status = clGetDeviceIDs(platforms[i], device_type, 0, NULL, &numDevices);
        //allocate enough space for each devices
        cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
        clGetDeviceIDs(platforms[i], device_type, numDevices, devices, NULL);

        if (status == CL_SUCCESS) {
            return devices[0];
        }
    }
}


// ## You may add your own initialization routines here ##
void init() {
    FILE* fp;
    char* source_str;
    size_t source_size;

    fp = fopen("parallel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* ^^^^ ---- Physics Engine ^^^^ ---- */
    // set CPU to do physics
    cl_device_id cpu_device = getDeviceID(CL_DEVICE_TYPE_CPU);
    chk(status, "clGetDeviceIDs");

    // Create context 
    physics_context = clCreateContext(NULL, 1, &cpu_device, NULL, NULL, &status);
    chk(status, "clCreateContext");

    // Create command queue 
    physics_queue = clCreateCommandQueue(physics_context, cpu_device, 0, &status);
    chk(status, "clCreateCommandQueue");

    // Create buffer for satellites 
    physics_satellites_buffer = clCreateBuffer(physics_context, CL_MEM_USE_HOST_PTR,
        SATELLITE_SIZE, satellites, &status);
    chk(status, "clCreateBuffer");

    // Create program 
    physics_program = clCreateProgramWithSource(physics_context, 1, (const char**)&source_str,
        (const size_t*)&source_size, &status);
    chk(status, "clCreateProgramWithSource");

    // Build the program 
    status = clBuildProgram(physics_program, 1, &cpu_device, "-cl-fast-relaxed-math", NULL, NULL);
    chk(status, "clBuildProgram");

    // Create kernel
    physics_kernel = clCreateKernel(physics_program, "physics_engine_kernel", &status);

    // Set argument 
    status = clSetKernelArg(physics_kernel, 0, sizeof(cl_mem), (void*)&physics_satellites_buffer);
    chk(status, "clCreateKernel");

    /* ^^^^ ---- Graphics Engine ^^^^ ---- */
    // set GPU for graphics
    cl_device_id gpu_device = getDeviceID(CL_DEVICE_TYPE_GPU);
    chk(status, "clGetDeviceIDs");

    // Create context 
    graphics_context = clCreateContext(NULL, 1, &gpu_device, NULL, NULL, &status);

    // Create command queue 
    graphics_queue = clCreateCommandQueue(graphics_context, gpu_device, 0, &status);
    chk(status, "clCreateCommandQueue");

    // Create buffer for satellites and pixels 
    graphics_satellites_buffer = clCreateBuffer(graphics_context, CL_MEM_USE_HOST_PTR,
        SATELLITE_SIZE, satellites, &status);
    chk(status, "clCreateBuffer");

    pixels_buffer = clCreateBuffer(graphics_context, CL_MEM_USE_HOST_PTR,
        PIXEL_SIZE, pixels, &status);
    chk(status, "clCreateBuffer");

    // Create program 
    graphics_program = clCreateProgramWithSource(graphics_context, 1, (const char**)&source_str,
        (const size_t*)&source_size, &status);
    chk(status, "clCreateProgramWithSource");

    // Build the program 
    status = clBuildProgram(graphics_program, 1, &gpu_device, "-cl-fast-relaxed-math", NULL, NULL);
    chk(status, "clBuildProgram");

    // Create kernel
    graphics_kernel = clCreateKernel(graphics_program, "graphics_engine_kernel", &status);
    chk(status, "clCreateKernel");

    // Set arguments 
    status = clSetKernelArg(graphics_kernel, 0, sizeof(cl_mem), (void*)&graphics_satellites_buffer);
    status |= clSetKernelArg(graphics_kernel, 1, sizeof(cl_mem), (void*)&pixels_buffer);
    chk(status, "clSetKernelArg");

    // Set workgroup size 
    printf("Enter WG horizontal size: ");
    scanf("%zu", &local[0]);

    printf("Enter WG vertical size: ");
    scanf("%zu", &local[1]);
}


// ## You are asked to make this code parallel ##
// Physics engine loop. (This is called once a frame before graphics engine) 
// Moves the satelites based on gravity
// This is done multiple times in a frame because the Euler integration 
// is not accurate enough to be done only once
void parallelPhysicsEngine() {
    // Total number of satellites
    size_t global = SATELLITE_COUNT;

    // Execute the kernel
    status = clEnqueueNDRangeKernel(physics_queue, physics_kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    chk(status, "clEnqueueNDRange");

    clFinish(physics_queue);
}


// ## You are asked to make this code parallel ##
// Rendering loop (This is called once a frame after physics engine) 
// Decides the color for each pixel.
void parallelGraphicsEngine() {
    // Total number of pixels
    size_t global[2] = { WINDOW_HEIGHT, WINDOW_WIDTH };

    // Copy the input to the device
    status = clEnqueueWriteBuffer(graphics_queue, graphics_satellites_buffer,
        CL_TRUE, 0, SATELLITE_SIZE, satellites, 0, NULL, NULL);
    chk(status, "clEnqueueWriteBuffer");

    // Execute the kernel
    status = clEnqueueNDRangeKernel(graphics_queue, graphics_kernel,
        2, NULL, global, local, 0, NULL, NULL);
    chk(status, "clEnqueueNDRange");

    // Read data back to the host
    status = clEnqueueReadBuffer(graphics_queue, pixels_buffer, CL_TRUE,
        0, PIXEL_SIZE, pixels, 0, NULL, NULL);
    chk(status, "clEnqueueReadBuffer");

    clFlush(graphics_queue);
    clFinish(graphics_queue);
}

// ## You may add your own destrcution routines here ##
void destroy() {
    clReleaseKernel(physics_kernel);
    clReleaseKernel(graphics_kernel);

    clReleaseProgram(physics_program);
    clReleaseProgram(graphics_program);

    clReleaseCommandQueue(physics_queue);
    clReleaseCommandQueue(graphics_queue);

    clReleaseMemObject(physics_satellites_buffer);
    clReleaseMemObject(graphics_satellites_buffer);
    clReleaseMemObject(pixels_buffer);

    clReleaseContext(physics_context);
    clReleaseContext(graphics_context);
}







////////////////////////////////////////////////
// い TO NOT EDIT ANYTHING AFTER THIS LINE い //
////////////////////////////////////////////////

// い DO NOT EDIT THIS FUNCTION い
// Sequential rendering loop used for finding errors
void sequentialGraphicsEngine() {

    // Graphics pixel loop
    for (int i = 0; i < SIZE; ++i) {

        // Row wise ordering
        floatvector pixel = { .x = i % WINDOW_WIDTH, .y = i / WINDOW_WIDTH };

        // This color is used for coloring the pixel
        color renderColor = { .red = 0.f, .green = 0.f, .blue = 0.f };

        // Find closest satellite
        float shortestDistance = INFINITY;

        float weights = 0.f;
        int hitsSatellite = 0;

        // First Graphics satellite loop: Find the closest satellite.
        for (int j = 0; j < SATELLITE_COUNT; ++j) {
            floatvector difference = { .x = pixel.x - satellites[j].position.x,
                                      .y = pixel.y - satellites[j].position.y };
            float distance = sqrt(difference.x * difference.x +
                difference.y * difference.y);

            if (distance < SATELLITE_RADIUS) {
                renderColor.red = 1.0f;
                renderColor.green = 1.0f;
                renderColor.blue = 1.0f;
                hitsSatellite = 1;
                break;
            }
            else {
                float weight = 1.0f / (distance * distance * distance * distance);
                weights += weight;
                if (distance < shortestDistance) {
                    shortestDistance = distance;
                    renderColor = satellites[j].identifier;
                }
            }
        }

        // Second graphics loop: Calculate the color based on distance to every satellite.
        if (!hitsSatellite) {
            for (int j = 0; j < SATELLITE_COUNT; ++j) {
                floatvector difference = { .x = pixel.x - satellites[j].position.x,
                                          .y = pixel.y - satellites[j].position.y };
                float dist2 = (difference.x * difference.x +
                    difference.y * difference.y);
                float weight = 1.0f / (dist2 * dist2);

                renderColor.red += (satellites[j].identifier.red *
                    weight / weights) * 3.0f;

                renderColor.green += (satellites[j].identifier.green *
                    weight / weights) * 3.0f;

                renderColor.blue += (satellites[j].identifier.blue *
                    weight / weights) * 3.0f;
            }
        }
        correctPixels[i] = renderColor;
    }
}

void sequentialPhysicsEngine(satellite* s) {

    // double precision required for accumulation inside this routine,
    // but float storage is ok outside these loops.
    doublevector tmpPosition[SATELLITE_COUNT];
    doublevector tmpVelocity[SATELLITE_COUNT];

    for (int i = 0; i < SATELLITE_COUNT; ++i) {
        tmpPosition[i].x = s[i].position.x;
        tmpPosition[i].y = s[i].position.y;
        tmpVelocity[i].x = s[i].velocity.x;
        tmpVelocity[i].y = s[i].velocity.y;
    }

    // Physics iteration loop
    for (int physicsUpdateIndex = 0;
        physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
        ++physicsUpdateIndex) {

        // Physics satellite loop
        for (int i = 0; i < SATELLITE_COUNT; ++i) {

            // Distance to the blackhole
            // (bit ugly code because C-struct cannot have member functions)
            doublevector positionToBlackHole = { .x = tmpPosition[i].x -
               HORIZONTAL_CENTER, .y = tmpPosition[i].y - VERTICAL_CENTER };
            double distToBlackHoleSquared =
                positionToBlackHole.x * positionToBlackHole.x +
                positionToBlackHole.y * positionToBlackHole.y;
            double distToBlackHole = sqrt(distToBlackHoleSquared);

            // Gravity force
            doublevector normalizedDirection = {
               .x = positionToBlackHole.x / distToBlackHole,
               .y = positionToBlackHole.y / distToBlackHole };
            double accumulation = GRAVITY / distToBlackHoleSquared;

            // Delta time is used to make velocity same despite different FPS
            // Update velocity based on force
            tmpVelocity[i].x -= accumulation * normalizedDirection.x *
                DELTATIME / PHYSICSUPDATESPERFRAME;
            tmpVelocity[i].y -= accumulation * normalizedDirection.y *
                DELTATIME / PHYSICSUPDATESPERFRAME;

            // Update position based on velocity
            tmpPosition[i].x +=
                tmpVelocity[i].x * DELTATIME / PHYSICSUPDATESPERFRAME;
            tmpPosition[i].y +=
                tmpVelocity[i].y * DELTATIME / PHYSICSUPDATESPERFRAME;
        }
    }

    // double precision required for accumulation inside this routine,
    // but float storage is ok outside these loops.
    // copy back the float storage.
    for (int i = 0; i < SATELLITE_COUNT; ++i) {
        s[i].position.x = tmpPosition[i].x;
        s[i].position.y = tmpPosition[i].y;
        s[i].velocity.x = tmpVelocity[i].x;
        s[i].velocity.y = tmpVelocity[i].y;
    }
}

// Just some value that barely passes for OpenCL example program
#define ALLOWED_FP_ERROR 0.08
// い DO NOT EDIT THIS FUNCTION い
void errorCheck() {
    for (unsigned int i = 0; i < SIZE; ++i) {
        if (fabs(correctPixels[i].red - pixels[i].red) > ALLOWED_FP_ERROR ||
            fabs(correctPixels[i].green - pixels[i].green) > ALLOWED_FP_ERROR ||
            fabs(correctPixels[i].blue - pixels[i].blue) > ALLOWED_FP_ERROR) {
            printf("Buggy pixel at (x=%i, y=%i). Press enter to continue.\n", i % WINDOW_WIDTH, i / WINDOW_WIDTH);
            getchar();
            return;
        }
    }
    printf("Error check passed!\n");
}

// い DO NOT EDIT THIS FUNCTION い
void compute(void) {
    int timeSinceStart = glutGet(GLUT_ELAPSED_TIME);
    previousFrameTimeSinceStart = timeSinceStart;

    // Error check during first frames
    if (frameNumber < 2) {
        memcpy(backupSatelites, satellites, sizeof(satellite) * SATELLITE_COUNT);
        sequentialPhysicsEngine(backupSatelites);
    }
    parallelPhysicsEngine();
    if (frameNumber < 2) {
        for (int i = 0; i < SATELLITE_COUNT; i++) {
            if (memcmp(&satellites[i], &backupSatelites[i], sizeof(satellite))) {
                printf("Incorrect satellite data of satellite: %d\n", i);
                getchar();
            }
        }
    }

    int satelliteMovementMoment = glutGet(GLUT_ELAPSED_TIME);
    int satelliteMovementTime = satelliteMovementMoment - timeSinceStart;

    // Decides the colors for the pixels
    parallelGraphicsEngine();

    int pixelColoringMoment = glutGet(GLUT_ELAPSED_TIME);
    int pixelColoringTime = pixelColoringMoment - satelliteMovementMoment;

    // Sequential code is used to check possible errors in the parallel version
    if (frameNumber < 2) {
        sequentialGraphicsEngine();
        errorCheck();
    }

    int finishTime = glutGet(GLUT_ELAPSED_TIME);
    // Print timings
    int totalTime = finishTime - previousFinishTime;
    previousFinishTime = finishTime;

    printf("Total frametime: %ims, satellite moving: %ims, space coloring: %ims.\n",
        totalTime, satelliteMovementTime, pixelColoringTime);

    // Render the frame
    glutPostRedisplay();
}

// い DO NOT EDIT THIS FUNCTION い
// Probably not the best random number generator
float randomNumber(float min, float max) {
    return (rand() * (max - min) / RAND_MAX) + min;
}

// DO NOT EDIT THIS FUNCTION
void fixedInit(unsigned int seed) {

    if (seed != 0) {
        srand(seed);
    }

    // Init pixel buffer which is rendered to the widow
    pixels = (color*)malloc(sizeof(color) * SIZE);

    // Init pixel buffer which is used for error checking
    correctPixels = (color*)malloc(sizeof(color) * SIZE);

    backupSatelites = (satellite*)malloc(sizeof(satellite) * SATELLITE_COUNT);


    // Init satellites buffer which are moving in the space
    satellites = (satellite*)malloc(sizeof(satellite) * SATELLITE_COUNT);

    // Create random satellites
    for (int i = 0; i < SATELLITE_COUNT; ++i) {

        // Random reddish color
        color id = { .red = randomNumber(0.f, 0.15f) + 0.1f,
                    .green = randomNumber(0.f, 0.14f) + 0.0f,
                    .blue = randomNumber(0.f, 0.16f) + 0.0f };

        // Random position with margins to borders
        floatvector initialPosition = { .x = HORIZONTAL_CENTER - randomNumber(50, 320),
                                .y = VERTICAL_CENTER - randomNumber(50, 320) };
        initialPosition.x = (i / 2 % 2 == 0) ?
            initialPosition.x : WINDOW_WIDTH - initialPosition.x;
        initialPosition.y = (i < SATELLITE_COUNT / 2) ?
            initialPosition.y : WINDOW_HEIGHT - initialPosition.y;

        // Randomize velocity tangential to the balck hole
        floatvector positionToBlackHole = { .x = initialPosition.x - HORIZONTAL_CENTER,
                                      .y = initialPosition.y - VERTICAL_CENTER };
        float distance = (0.06 + randomNumber(-0.01f, 0.01f)) /
            sqrt(positionToBlackHole.x * positionToBlackHole.x +
                positionToBlackHole.y * positionToBlackHole.y);
        floatvector initialVelocity = { .x = distance * -positionToBlackHole.y,
                                  .y = distance * positionToBlackHole.x };

        // Every other orbits clockwise
        if (i % 2 == 0) {
            initialVelocity.x = -initialVelocity.x;
            initialVelocity.y = -initialVelocity.y;
        }

        satellite tmpSatelite = { .identifier = id, .position = initialPosition,
                                .velocity = initialVelocity };
        satellites[i] = tmpSatelite;
    }
}

// い DO NOT EDIT THIS FUNCTION い
void fixedDestroy(void) {
    destroy();

    free(pixels);
    free(correctPixels);
    free(satellites);

    if (seed != 0) {
        printf("Used seed: %i\n", seed);
    }
}

// い DO NOT EDIT THIS FUNCTION い
// Renders pixels-buffer to the window 
void render(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_FLOAT, pixels);
    glutSwapBuffers();
    frameNumber++;
}

// DO NOT EDIT THIS FUNCTION
// Inits glut and start mainloop
int main(int argc, char** argv) {

    if (argc > 1) {
        seed = atoi(argv[1]);
        printf("Using seed: %i\n", seed);
    }

    // Init glut window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow("Parallelization excercise");
    glutDisplayFunc(render);
    atexit(fixedDestroy);
    previousFrameTimeSinceStart = glutGet(GLUT_ELAPSED_TIME);
    previousFinishTime = glutGet(GLUT_ELAPSED_TIME);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    fixedInit(seed);
    init();

    // compute-function is called when everythin from last frame is ready
    glutIdleFunc(compute);

    // Start main loop
    glutMainLoop();
}