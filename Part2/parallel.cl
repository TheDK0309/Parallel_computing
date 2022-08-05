#include "parallel.h"
__kernel void physics_engine_kernel(__global satellite* satellites) {

    // Copy the input image to the device
    size_t globalId = get_global_id(0);

    // double precision required for accumulation inside this routine,
    // but float storage is ok outside these loops.	
    __private doublevector tmpPosition;
    __private doublevector tmpVelocity;
    tmpPosition.x = satellites[globalId].position.x;
    tmpPosition.y = satellites[globalId].position.y;
    tmpVelocity.x = satellites[globalId].velocity.x;
    tmpVelocity.y = satellites[globalId].velocity.y;

    // Physics iteration loop
    for (int physicsUpdateIndex = 0;
        physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
        ++physicsUpdateIndex) {

        // Distance to the blackhole (bit ugly code because C-struct cannot have member functions)
        doublevector positionToBlackHole = { .x = tmpPosition.x -
            HORIZONTAL_CENTER, .y = tmpPosition.y - VERTICAL_CENTER };
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
        tmpVelocity.x -= accumulation * normalizedDirection.x *
            DELTATIME / PHYSICSUPDATESPERFRAME;
        tmpVelocity.y -= accumulation * normalizedDirection.y *
            DELTATIME / PHYSICSUPDATESPERFRAME;

        // Update position based on velocity
        tmpPosition.x +=
            tmpVelocity.x * DELTATIME / PHYSICSUPDATESPERFRAME;
        tmpPosition.y +=
            tmpVelocity.y * DELTATIME / PHYSICSUPDATESPERFRAME;

    }

    // double precision required for accumulation inside this routine,
    // but float storage is ok outside these loops.
    // copy back the float storage.
    satellites[globalId].position.x = tmpPosition.x;
    satellites[globalId].position.y = tmpPosition.y;
    satellites[globalId].velocity.x = tmpVelocity.x;
    satellites[globalId].velocity.y = tmpVelocity.y;

}

__kernel void graphics_engine_kernel(__global satellite* satellites,
    __global color* pixels) {

    //Work-item gets its index within index space
    size_t ix = get_global_id(1);
    size_t iy = get_global_id(0);

    // Row wise ordering
    __private floatvector pixel = { .x = ix , .y = iy };

    // This color is used for coloring the pixel
    __private color renderColor = { .red = 0.f, .green = 0.f, .blue = 0.f };

    __private float red_temp = 0.0f;
    __private float blue_temp = 0.0f;
    __private float green_temp = 0.0f;

    // Find closest satelite
    float shortestDistance = INFINITY;

    float weights = 0.f;
    int hitsSatellite = 0;

    // First Graphics satellite loop: Find the closest satellite.
    for (int j = 0; j < SATELLITE_COUNT; ++j) {

        floatvector difference = { .x = pixel.x - satellites[j].position.x,
                                    .y = pixel.y - satellites[j].position.y };
        float distance = sqrt(difference.x * difference.x + difference.y * difference.y);

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
            red_temp += satellites[j].identifier.red * weight;
            green_temp += satellites[j].identifier.green * weight;
            blue_temp += satellites[j].identifier.blue * weight;
        }
    }
    // Second graphics loop: Calculate the color based on distance to every satellite.
    if (!hitsSatellite) {
        renderColor.red += (red_temp / weights) * 3.0f;
        renderColor.green += (green_temp / weights) * 3.0f;
        renderColor.blue += (blue_temp / weights) * 3.0f;
    }
    pixels[ix + WINDOW_WIDTH * iy] = renderColor;
}
