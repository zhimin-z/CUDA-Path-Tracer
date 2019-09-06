#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include <glm/gtc/matrix_inverse.hpp>
#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
# endif
}


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}


static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;
// TODO: static variables for device memory, scene/camera info, etc
// ...
static Ray *dev_ray = NULL;
static int *dev_bool = NULL;
static Ray *dev_compacted = NULL;
static Material *dev_m;
static Geom *dev_geo = NULL;
static Geom *dev_geoms;
static glm::vec3 *dev_test = NULL;
static glm::vec3 *dev_colormap = NULL;
static int * dev_combool = NULL;
static int * dev_resultint = NULL;
static Ray * dev_resultray = NULL;
static Ray *camera_ray = NULL;
static Ray *dev_compatedRay = NULL;
static bool* dev_terminate = NULL;
static int * dev_newcombinedSumdata = NULL;
static Ray *dev_compactResult=NULL;

__device__ void depth_of_field(Camera cmr,int iter, int index,Ray &ray,int x,int y){
	glm::vec3 horizontal, middle, vertical;

	glm::vec3 F = glm::normalize(cmr.view);
	glm::vec3 R = glm::normalize(glm::cross(F, cmr.up));
	glm::vec3 U = glm::normalize(glm::cross(R, F));
	float len = glm::length(cmr.view);
	int width = cmr.resolution.x;
	int height = cmr.resolution.y;
	float alpha = cmr.fov.y*PI / 180.f;
	glm::vec3 V = U*len *tan(alpha);
	float temp = width*1.0 / (height*1.0);
	glm::vec3 H = temp*glm::length(F)*R;
	//camera componet(EYE:0,4,7)
	float lens_radius = 2.0f;
	float focal_distance = 2.f;//aperon focus
	// step1.Sample a random point on the lense 
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 1);
	thrust::uniform_real_distribution<float> u01(-1, 1);
	//a concentric sample disk
	float theta = PI * u01(rng);
	float dx = lens_radius*cosf(theta);
	float dy = lens_radius * sinf(theta);
	glm::vec2 point = glm::vec2(dx, dy);
	glm::vec3 position = cmr.position + glm::vec3(dx, dy, 0);
	float ft = focal_distance / cmr.view.z;//cmr.position.z;
	thrust::uniform_real_distribution<float> uee(-EPSILON, EPSILON);

	float offset = uee(rng);
	float xx, yy;
	xx = 2.0* x / width - 1.0 + offset;
	yy = 1.0 - 2.0* y / height + offset;

	glm::vec3 point_pos = cmr.view + position + xx*H + yy*V;
	//glm::vec3 point_focus = cmr.view*ft + position;
	//ray.origin = cmr.position + (dx * R + dy * U);;
	ray.origin = position;
	ray.direction = glm::normalize(point_pos - ray.origin);

	

}
__global__ void SetDevRay(Camera cmr, Ray  *dev_ray, int iter){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cmr.resolution.x && y < cmr.resolution.y) {
		int index = x + (y * cmr.resolution.x);
		glm::vec3 F = glm::normalize(cmr.view);
		glm::vec3 R = glm::normalize(glm::cross(F, cmr.up));
		glm::vec3 U = glm::normalize(glm::cross(R, F));
		float len = glm::length(cmr.view);
		int width = cmr.resolution.x;
		int height = cmr.resolution.y;
		float alpha = cmr.fov.y*PI / 180.f;
		glm::vec3 V = U*len *tan(alpha);
		float temp = width*1.0 / (height*1.0);
		glm::vec3 H = temp*glm::length(F)*R;
		float xx;
		float yy;
		//jittering rays within an aperture
	

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 1);
		thrust::uniform_real_distribution<float> uee(-EPSILON, EPSILON);
	
		float offset = uee(rng);
		
		xx = 2.0* x / width - 1.0 + offset;
		yy = 1.0 - 2.0* y / height + offset;
	
		glm::vec3 point_pos = cmr.view + cmr.position + xx*H + yy*V;//glm::vec3 M_ = cmr.position + C_;
		//screen point to world point
		dev_ray[index].direction = glm::normalize(point_pos - cmr.position );
		dev_ray[index].origin = cmr.position;
		dev_ray[index].terminate = false;
		dev_ray[index].hitcolor = WHITE;
		///////Camera cmr,int iter, int index,Ray &ray,int x,int y
		depth_of_field(cmr, iter, index, dev_ray[index],x,y);
	}
}

void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = (cam.resolution.x) * (cam.resolution.y);

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
	// TODO: initialize the above static variables added above
	///////////////////
	cudaMalloc(&dev_compactResult, pixelcount * sizeof(Ray));
	cudaMemset(dev_compactResult, 0, pixelcount * sizeof(Ray));

	int geoSize = hst_scene->geoms.size()*sizeof(Geom);
	Geom * hst_geoms = (Geom *)malloc(geoSize);
	std::copy(hst_scene->geoms.begin(), hst_scene->geoms.end(), hst_geoms);
	cudaMalloc((void**)&dev_geoms, geoSize);
	cudaMemcpy(dev_geoms, hst_geoms, geoSize, cudaMemcpyHostToDevice);

	//////////////
	///////
	//camera_ray = new Ray[pixelcount];

	int nG = hst_scene->geoms.size();
	int nM = hst_scene->materials.size();

	Material * mm = new Material[nM];
	Geom *gg = new Geom[nG];
	for (int i = 0; i < nM; i++){
		mm[i] = hst_scene->materials[i];
	}
	for (int i = 0; i < nG; i++){
		gg[i] = hst_scene->geoms[i];
	}
	cudaMalloc((void**)&dev_m, nM*sizeof(Material));//n*sizeof(M)
	cudaMemcpy(dev_m, mm, nM*sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_geo, nG*sizeof(Geom));
	cudaMemcpy(dev_geo, gg, nG*sizeof(Geom), cudaMemcpyHostToDevice);

	// TODO: initialize the above static variables added above
	cudaMalloc(&dev_ray, pixelcount * sizeof(Ray));
	cudaMemset(dev_ray, 0, pixelcount * sizeof(Ray));

	cudaMalloc(&dev_colormap, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_colormap, 0, pixelcount * sizeof(glm::vec3));

	checkCUDAError("pathtraceInit");

}

__global__ void generateIamge(Camera cam, int iter, glm::vec3 *image, Ray *r) {

	/*int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	//if (id < (cam.resolution.x * cam.resolution.y)) {
	//if ()*/
	/*int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
	int id = x + (y * cam.resolution.x);
	if (r[id].terminate)
	image[id] += r[id].hitcolor;
	}*/
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (id < cam.resolution.x *cam.resolution.y) {

		if (r[id].terminate){
			image[id] += r[id].hitcolor;
			//r[id].hitcolor = glm::vec3();
		}
	}
}
__device__ void directlightcheking(Ray &r, Geom *dev_geom, int nG, const int light, Material m_light, int iter, int id){
	Ray light_ray;
	//find random place
	glm::vec3 point;
	float result[3];
	if (dev_geom[light].type == CUBE){
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, id, 1);
		thrust::uniform_real_distribution <float> side(0, 6);
		thrust::uniform_real_distribution <float> x01(-0.5, 0.5);
		thrust::uniform_real_distribution <float> y01(-0.5, 0.5);
		int s = (int)side(rng);
		int c = s % 3;
		result[c] = s > 2.0 ? 1.f : 0.f;
		result[(c + 1) % 3] = x01(rng);
		result[(c + 2) % 3] = y01(rng);
		point = glm::vec3(result[0], result[1], result[2]);

		glm::vec4 pt = glm::vec4(point, 1)* dev_geom[light].transform;
		point = glm::vec3(pt[0], pt[1], pt[2]);
	}

	if (dev_geom[light].type == SPHERE){
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, id, 1);
		thrust::uniform_real_distribution <float> u01(0, 1);
		thrust::uniform_real_distribution <float> v01(0, 1);

		//float radius = dev_geom[light].scale
		float u = u01(rng);
		float v = v01(rng);
		float theta = 2 * PI * u;
		float phi = acos(2 * v - 1);
		float x = 0.5 * sin(phi) * cos(theta);
		float y = 0.5 * sin(phi) * sin(theta);
		float z = 0.5 * cos(phi);

		point = glm::vec3(x, y, z);
		point *= dev_geom[light].translation;
	}

	light_ray.direction = point - r.origin;
	light_ray.origin = r.origin;
	float t;
	glm::vec3 intersectionPoint;
	glm::vec3 normal;
	bool outside;
	for (int i = 0; i < nG; i++){
		if (i != light){
			if (dev_geom[i].type == SPHERE){
				t = sphereIntersectionTest(dev_geom[i], r, intersectionPoint, normal, outside);
			}
			if (dev_geom[i].type == CUBE){
				t = boxIntersectionTest(dev_geom[i], r, intersectionPoint, normal, outside);
			}
		}
		if (i == light)t = -1;
	}

	if (t < 0){
		r.hitcolor *= m_light.emittance*m_light.color;
	}
}
__global__ void raytracing(int frame, Ray *r, int CurrentRayNumber, Geom *dev_geom, int nG, int nM, Material *dev_m, int iter, int traced, int currentd)
{
	//glm::vec3 s_translate = glm::vec3(1, 0, 0);
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < CurrentRayNumber){
		if (!r[id].terminate){
			int go_sphere = 2;
			glm::vec3 normal;
			glm::vec3 intersectionPoint;
			bool outside = false;
			float t = -2.0;
			int mId = 0;
			float mint = 30000;
			bool mark = false;
			Material m_light;
			int light;
			glm::vec3 F_normal;
			glm::vec3 F_intersectionPoint;
			bool F_outside = false;
			float F_t = -1.0;
			for (int i = 0; i < nM; i++){
				if (dev_m[i].emittance>0)//light
				{
					light = i;
					m_light = dev_m[i];
				}
			}

			for (int i = 0; i < nG; i++){
				int ll = 0;
				if (dev_geom[i].type == SPHERE){
					ll++;
					if (ll == go_sphere){
						t = sphereIntersectionTest(dev_geom[i], r[id], intersectionPoint, normal, outside, 0, frame);
					}
					else
						t = sphereIntersectionTest(dev_geom[i], r[id], intersectionPoint, normal, outside, 0, frame);
				}
				if (dev_geom[i].type == CUBE){
					t = boxIntersectionTest(dev_geom[i], r[id], intersectionPoint, normal, outside);
				}
				if (t<0)continue;
				if (t > 0 && t < mint){//if I want to find the nearest intersect object
					mId = dev_geom[i].materialid;
					mint = t;
					mark = true;
					F_intersectionPoint = intersectionPoint;
					F_normal = normal;
					F_outside = outside;
					F_t = t;

					thrust::default_random_engine rng = makeSeededRandomEngine(iter, id, currentd);
					thrust::uniform_real_distribution<float> uee(0, EPSILON * 10);
					float offset = uee(rng);
					//along the normal
					F_intersectionPoint += offset*F_normal;
				}
			}
			if (F_t < 0)//hit nothing terminate
			{
				r[id].terminate = true;
				r[id].hitcolor *= BLACK;

			}
			else if (F_t > 0)//hit something
			{
				glm::vec3 emmited_c;
				glm::vec3 color;
				Material m = dev_m[mId];//if (mId<0 || mId>nM):erro
				if (m.emittance > 0){//***if hit light
					r[id].terminate = true;
					r[id].hitcolor *= m.emittance*m.color;
				}
				else{
					if (currentd == traced - 1){
						r[id].terminate = true;
						directlightcheking(r[id], dev_geom, nG, light, m_light, iter, id);
					}
					thrust::default_random_engine rng = makeSeededRandomEngine(iter, id, currentd);
					scatterRay(r[id], color, F_intersectionPoint, F_normal, outside, emmited_c, m, rng);
					//r[id].hitcolor *= color;
				}
			}
		}
	}
}

__global__ void MapBool(int *dev_bool,  Ray *dev_ray, int n) {
	
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < n){
		if (dev_ray[id].terminate)dev_bool[id] = 0;
		else dev_bool[id] = 1;
	}
}


__global__ void kernScatter(int n, Ray *odata, Ray *idata, const int *bools, const int *indices)
{
	int k = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (k < n){
		if (bools[k] ){
			int t = indices[k];
			odata[t] = idata[k];
		}
	}
}
__global__ void EfficientScan(int n, int *outdata, const int *indata){
//current size
	//Example 39-2 in Gems
	extern __shared__ int temp[];//stores the updated bool

	//int id = (blockIdx.x * blockDim.x) + threadIdx.x;//tid
	int thid = threadIdx.x;
	int offset = 1;
	int blocks = blockIdx.x * blockDim.x * 2;

	temp[2 * thid] = indata[2 * thid + blocks];
	temp[2 * thid + 1] = indata[2 * thid + blocks * 2 + 1];
	//...do in block maybe..
	//	int n = blockDim.x * 2;
	for (int d = n >> 1; d > 0; d >>= 1) //d = 0 to log2 n – 1
	{
		__syncthreads();
		if (thid < d){
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thid == 0)						//clear the last element
		temp[n - 1] = 0;

	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)  {
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	outdata[2 * thid + blocks] = indata[2 * thid];
	outdata[2 * thid + blocks + 1] = indata[2 * thid + 1];
}

__global__ void BlockSums(int n, int *odata, const int *idata) {
	 	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		odata[index] = idata[(index + 1) * 128 - 1];
}



__global__ void BlcockIncrement(int n, int *dev_data, const int *increments) {
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	//add back to each block
	if (id < n){
	 // new[0],new[1]...new[127] =new[0],new[1]...new[127]+increment[0];
	//  new[128],new[129]...new[255] =new[128],new[129]..new[255]+increment[1];
		dev_data[id] +=   increments[blockIdx.x];
	}
}




int StreamCompact(Ray *dev_ray, Ray *result ,int raynumber){

	//step1.compute temp bool ray
	cudaMalloc((void**)&dev_bool, raynumber * sizeof(int));
	cudaMemset(dev_bool, 0, raynumber * sizeof(int));
	cudaMalloc((void**)&dev_combool, raynumber * sizeof(int));
	cudaMemset(dev_combool, 0, raynumber * sizeof(int));
	dim3 GridSize = (raynumber + 128 - 1) / 128;
	dim3 BlockSize = 128;
	int *s=new int[raynumber];
	s[0] = -1;

	//const dim3 blockSize2d(8, 8);
//	const dim3 blocksPerGrid2d(64)
	
//	<<blocksPerGrid2d, blockSize2d>>
//	mapBool(int *dev_bool,  Ray *dev_ray, int n)
	MapBool <<< GridSize, BlockSize >> >(dev_bool, dev_ray, raynumber);
	cudaMemcpy(s, dev_bool, raynumber*sizeof(int), cudaMemcpyDeviceToHost);
	//checkCUDAError("mapbool");

	//************************************/
	//step2.scan Scan(raynumber, dev_compactbool, dev_bool);
	dim3 dim3_p0 = GridSize;
	int int_p0 = raynumber + 128 - 1 / 128;
	cudaMalloc(&dev_newcombinedSumdata, sizeof(int)* 128);
	while (int_p0 >= 1){
	int n = (int_p0 + 128 - 1) / 128;
		dim3 scanGridSize = (int_p0 + 128 - 1) / 128;
		dim3 scanBlockSize = 128;// (?)
		
		//cudaMalloc(&dev_result, sizeof(int)* 128);
		//EfficientScan(int * indata, int *outdata, int n)
		EfficientScan << <scanGridSize, scanBlockSize >> >(int_p0,dev_combool, dev_bool);
		checkCUDAError("scan");
		//step4.write total sum of each block to  a new array.
		//return the sum of each block
		BlockSums << <scanGridSize, scanBlockSize >> >(n, dev_newcombinedSumdata,dev_combool);
		checkCUDAError("sum");
		//dim3 gBlockSum = p0;
		int int_p0 = int_p0 / 128;

		checkCUDAError("blockIncrement");
	}

	BlcockIncrement << < GridSize, 128 >> >(raynumber, dev_combool, dev_combool);
	//step3.scatter:kernScatter(int n, Ray *odata, const Ray *idata, const int *bools, const int *indices)
	kernScatter << <GridSize, BlockSize >> >(raynumber, result, dev_ray, dev_bool, dev_combool);
	checkCUDAError("kernScatter");

	int current_ray_number;
	cudaMemcpy(&current_ray_number,&(dev_combool[raynumber-1]), sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(dev_ray, dev_resultray, sizeof(Ray)*current_ray_number, cudaMemcpyDeviceToDevice);

	cudaFree(dev_bool);
	cudaFree(dev_combool);
	checkCUDAError("whatever");
	return current_ray_number;


}




void pathtrace(uchar4 *pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;


	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	const int &nG = hst_scene->geoms.size();
	const int &nM = hst_scene->materials.size();

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray is a (ray, color) pair, where color starts as the
	//     multiplicative identity, white = (1, 1, 1).
	//   * For debugging, you can output your ray directions as colors.
	// * For each depth:
	//   * Compute one new (ray, color) pair along each path (using scatterRay).
	//     Note that many rays will terminate by hitting a light or hitting
	//     nothing at all. You'll have to decide how to represent your path rays
	//     and how you'll mark terminated rays.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//       surface.
	//     * You can debug your ray-scene intersections by displaying various
	//       values as colors, e.g., the first surface normal, the first bounced
	//       ray direction, the first unlit material color, etc.
	//   * Add all of the terminated rays' results into the appropriate pixels.
	//   * Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	// * Finally, handle all of the paths that still haven't terminated.
	//   (Easy way is to make them black or background-colored.)

	// TODO: perform one iteration of path tracing
	//}//use 256 warp erro


	//SetDevRay << <blocksPerGrid2d, blockSize2d >> >(cam, dev_ray, iter);
	const dim3 Grid_256 = (pixelcount + 256 - 1) / 256;
	const dim3 B_256 = 256;
	const dim3 Grid_128 = (pixelcount + 128 - 1) / 128;
	const dim3 B_128 = 128;
	int current_ray = pixelcount;
	int test;
	cudaMalloc(&dev_ray, current_ray*sizeof(Ray));
	for (int i = 0; i < traceDepth; i++)
	{//remove the ray number from pixel count,so the Grid size needs to be changed
		dim3 CurGrid_128 = (current_ray + 128 - 1) / 128;
		if (!i)
		{
			current_ray = pixelcount;
			//cudaMalloc(&dev_ray, current_ray*sizeof(Ray));
			SetDevRay << <blocksPerGrid2d, blockSize2d >> >(cam, dev_ray, iter);//shoot ray first time&one time
		}
		//(Ray *r, int CurrentRayNumber, Geom *dev_geom, int nG, int nM, Material *dev_m, int iter, int traced, int currentd)
		raytracing << <CurGrid_128, B_128 >> >(frame, dev_ray, current_ray, dev_geo, nG, nM, dev_m, iter, traceDepth, i);
		//checkCUDAError("raytrace");
		//current_ray = StreamCompact(dev_ray, dev_compactResult, current_ray);
		//test=StreamCompact(dev_ray, dev_compactResult, current_ray);
		//cudaMemcpy(dev_ray, dev_compactResult, sizeof(Ray)*current_ray , cudaMemcpyDeviceToDevice);	
	}
//	printf("%d", test);
	generateIamge << <Grid_128, B_128 >> >(cam, iter, dev_image, dev_ray);
	cudaFree(dev_ray);
	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> >(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	// TODO: clean up the above static variables
	cudaFree(dev_ray);
	cudaFree(dev_compacted);
	cudaFree(dev_compactResult);
	cudaFree(dev_m);
	cudaFree(dev_geo);
	cudaFree(dev_colormap);

	checkCUDAError("pathtraceFree");
}
