#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
glm::vec3 normal, thrust::default_random_engine &rng) {//generate a random number
	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = sqrt(u01(rng)); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = u01(rng) * TWO_PI;

	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways:
 * - Always take a 50/50 split between a diffuse bounce and a specular bounce,
 *   but multiply the result of either one by 1/0.5 to cancel the 0.5 chance
 *   of it happening.
 * - Pick the split based on the intensity of each color, and multiply each
 *   branch result by the inverse of that branch's probability (same as above).

 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 *//*float calculateFresnelReflectionCoefficient(glm::vec3 direction, glm::vec3 normal, float indexOfRefraction) {
	 49 	float r0 = glm::pow((1.0f - indexOfRefraction) / (1.0f + indexOfRefraction), 2);
	 50 	return r0 + (1.0f - r0) * glm::pow(1.0f - glm::dot(normal, -direction), 5);
	 51
 }*/

__host__ __device__ float R_coeff(float theta, float n2){
	//air :n1=1; ice:1.3

	float n1 = 1.0; //n2 = 1.3;
	float R0 = pow((n1 - n2) / (n1 + n2), 2);
	float Rtheta = R0 + (1 - R0)*pow((1 - theta), 5);
	return Rtheta;
}
__host__ __device__ void reflectRay(Ray &r, glm::vec3 normal, glm::vec3 intersection, Material m){
	r.direction = glm::reflect(r.direction, normal);
	r.origin = intersection;
	r.hitcolor *= m.specular.color;
}

__host__ __device__
void scatterRay(Ray &ray, glm::vec3 &color, glm::vec3 intersect, glm::vec3 normal, bool outside, glm::vec3 emittedColor,
const Material &m, thrust::default_random_engine &rng) {

	if (m.hasRefractive){

		float theta = glm::dot(-ray.direction, normal);
		float Rtheta = R_coeff(theta, m.indexOfRefraction);
		thrust::uniform_real_distribution <float> u01(0, 1);
		glm::vec3 refract_ray, reflect_ray;
		if (u01(rng) <1-Rtheta)//refract
		{
			//refractRay(ray, normal, intersect, m, outside);
			float n = 0;
			if (outside) {
			//float 	
				float n = 1.0f / m.indexOfRefraction; // why I have to wirte the float agian???
			}
			
			float sintheta2 = glm::pow(n, 2) *  glm::pow(glm::dot(-normal, ray.direction), 2);
			if (1.0-sintheta2>=0){
				glm::vec3 r1 = ray.direction;
				glm::vec3 r2 = n*r1 - (n*glm::dot(r1, normal) + float(glm::sqrt(1.0-sintheta2)))*normal;
				ray.direction = r2;
				ray.origin = intersect + ray.direction * 0.0005f;//maybe because of the shadow acne it canot shows itselt...
				//so I add an offset but why there is the black edge... 
			}
			//else I assume they are absored
		
			}
			else{
				reflectRay(ray, normal, intersect, m);
			}
		}
	//}
       if (m.hasReflective&&(!m.hasRefractive)){
			if (m.specular.exponent > 1){
				reflectRay(ray, normal, intersect, m);//perfect mirror
			}
		}
	
		//diffuse
    if ((!m.hasReflective) && (!m.hasRefractive)){
			glm::vec3 diffuse_ray = calculateRandomDirectionInHemisphere(normal, rng);
			ray.direction = glm::normalize(diffuse_ray);
			ray.origin = intersect;
			ray.hitcolor *= m.color;
		}
		//else color 
	}
