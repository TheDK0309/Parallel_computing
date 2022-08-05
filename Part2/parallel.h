// These are used to decide the window size
#define WINDOW_HEIGHT 1024
#define WINDOW_WIDTH 1024

// The number of satelites can be changed to see how it affects performance.
// Benchmarks must be run with the original number of satellites
#define SATELLITE_COUNT 64

// These are used to control the satelite movement
#define SATELLITE_RADIUS 3.16f
#define MAX_VELOCITY 0.1f
#define GRAVITY 1.0f
#define DELTATIME 32
#define PHYSICSUPDATESPERFRAME 100000

// Some helpers to window size variables
#define SIZE WINDOW_WIDTH*WINDOW_HEIGHT
#define HORIZONTAL_CENTER (WINDOW_WIDTH / 2)
#define VERTICAL_CENTER (WINDOW_HEIGHT / 2)
#define CL_TARGET_OPENCL_VERSION 120

#define PIXEL_SIZE sizeof(color) * SIZE
#define SATELLITE_SIZE sizeof(satellite) * SATELLITE_COUNT
#define MAX_SOURCE_SIZE (0x100000)

// Stores 2D data like the coordinates
typedef struct {
	float x;
	float y;
} floatvector;

// Stores 2D data like the coordinates
typedef struct {
	double x;
	double y;
} doublevector;

// Stores rendered colors. Each float may vary from 0.0f ... 1.0f
typedef struct {
	float red;
	float green;
	float blue;
} color;

// Stores the satelite data, which fly around black hole in the space
typedef struct {
	color identifier;
	floatvector position;
	floatvector velocity;
} satellite;