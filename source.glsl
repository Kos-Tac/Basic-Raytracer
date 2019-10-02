const float FLT_MAX = 99999999999999999999999.99;
const int MAX_NB_BOUNCES = 3;

struct Material
{
	float Ka; //ambient
	float Kd; //diffuse
	float Ks; //specular
	float Kn; //specular power
};
    
//MISCELANOUS EFFECTS
    
//NO EFFECTS
//Vanilla mode activation parameters
bool vanilla = false;

//Blurr
//Activation parameter
bool blurr = false;

//Pseudo motion blurr (it is just a blurr)
//Activation parameter
bool motionBlurr = false;

bool firstLoop = true;
vec3 imgBuffer[8];
const int NB_MOTIONS_BUFFER = 4;
vec3 objectColForBlurr;

//Anti-Aliasing parameters
const int PIXEL_SAMPLING_GRID_SIZE = 8;
//Activation parameter for stochastic version or not (result is weird)
bool stoch = false;



// Camera parameters
vec3	cameraPos    = vec3(6, 4, -5);
const vec3	cameraTarget = vec3(3, 1, -8);
const float cameraFovY   = 80.0;			// NOTE: angle in degree

// Sphere parameters
const int      sphereId     = 1;
vec3  spherePos    = cameraTarget + vec3(0, 1, 2);
const float sphereRadius = 1.0;
const vec3 sphereCol 	 = vec3(1,0,0);
const Material sphereMat = Material(0.2,0.7,1.0,50.0);
const float transparency = 4.0;

// CSG Object Parameters
vec3           csgspherePos1    = cameraTarget + vec3(-1.125, 2, 0);
const float    csgsphereRadius1 = 1.4;
vec3           csgspherePos2    = cameraTarget + vec3(1.125, 2, 0);
const float    csgsphereRadius2 = 1.4;
vec3           csgspherePos3    = cameraTarget + vec3(0, 2.5, 0);
const float    csgsphereRadius3 = 0.5;
vec3           csgspherePos4    = cameraTarget + vec3(0, 2.4, -0.5);
const float    csgsphereRadius4 = 0.5;
const vec3     csgCol1 	        = vec3(1.0,0.5,0.0);//orange
const vec3     csgCol2	        = vec3(0.4,1.0,1.0);//light cyan
const vec3 	   csgCol3 	        = vec3(1.0,0.0,1.0);//magenta
const vec3     csgCol4 	        = vec3(5.0,5.0,0.0);//flashy yellow
const Material csgMat           = Material(0.2,1.0,0.1,90.0);
const int      csgId            = 3;



// Plane parameters
const vec3     planePos     = vec3(0, 0.1, 0);
const vec3     planeNormal  = vec3(0, 1.0, 0);
const vec3     planeCol1    = vec3(1.0);		// white
const vec3     planeCol2    = vec3(0.4);		// gray
const Material planeMat     = Material(0.2/*Ka*/, 1.0/*Kd*/, 0.2/*Ks*/,  5.0/*Kn*/);
const int      planeId      = 2;

// Sky parameters
const vec3     skyCl       = vec3(0,0,0);	
const int      skyId        = 0;


// Light(s) parameters
const vec3 ambiantCol = vec3(0,0,1);
const vec3 lightCol = vec3(1,1,1);
vec3 lightPos = vec3(8,10,-12);
float shadowFactor = 1.0;

const vec3  lightCol1 = vec3(1,1,1);
vec3        lightPos1 = vec3(8,10,-12);
const float lightPow1 = 0.8;
const vec3  lightCol2 = vec3(1,1,0.5);
vec3        lightPos2 = vec3(3,10,1);
const float lightPow2 = 0.5;

const int NB_LIGHTS = 2;


struct LightInfo{
    vec3 pos;
    vec3 col;
    float power;
};

LightInfo lights[NB_LIGHTS];


//-----------------------------------------------//





struct ShadeInfo
{
    vec3 shadedCol;
    float Ks;
};
    
    

void AnimateScene(float time)
{
    const float pi = 3.1415;
    const float rs  = 2.0;
    const float spr = 5.0;
    float as = 2.0 * pi * time / spr;

    spherePos = cameraTarget + rs * vec3(-sin(as), 0.0, cos(as)) + vec3(0,1,0);

    //lightPos += vec3(0, 10.5 + 9.5 * cos(time) - 10., 0);
    float targetDist = length(cameraTarget - cameraPos);
    cameraPos -= vec3(0, 0, targetDist);
    cameraPos += targetDist * vec3(min(sin(time),-0.1), max(sin(time*0.5),0.1), cos(time));

}



float rayPlane(vec3 rayPos, vec3 rayDir, vec3 planePos, vec3 planeNormal, out vec3 intersecPt, out vec3 normal)
{
    
 	  float den = dot(planeNormal, rayDir);
    
    if (abs(den) <= 0.000001)	
        return -1.0;			
        						
        					
    float t = dot(planeNormal, planePos - rayPos) / den;
    
   
    intersecPt = rayPos + t * rayDir;
    
  
    normal = -sign(den) * planeNormal;
    
    return t;
    
}

float raySphere(vec3 rayPos, vec3 rayDir, vec3 spherePos, float sphereRadius, out vec3 intersecS,out vec3 normalS)
{
    //Distance between the center of the sphere and the initial position of the ray
    vec3 diff = rayPos - spherePos;
    
    //Operations with dot lead to the following equation:
    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(diff  , rayDir);
    float c = dot(diff  , diff  ) - sphereRadius * sphereRadius;
    
    //Computing of the discriminant to find the 2 slutions t1 and t2 (points moving in time hence each position p of a point
    //corresponds to rayPos(bse position) + t*rayDir with t moving

    float dist = b*b - 4.0*a*c;
    
    if (dist >= 0.0){
        float sdi = sqrt(dist);
        float den = 1. / (2.0 * a);
        float t1  = (-b - sdi) * den;
        float t2  = (-b + sdi) * den;
        
        bool res = (t1 > 0.0) || (t2 > 0.0);

        if(t1 > 0.0){
            intersecS = rayPos + t1*rayDir;
            normalS = normalize(intersecS - spherePos);
            return t1;
        }
        else if (t2 > 0.0){
        	intersecS = rayPos + t2*rayDir;
            normalS = normalize(intersecS - spherePos);
            return t2;
        }
     
    }  
    return -1.0;
}


float raySphere2(vec3 rayPos, vec3 rayDir, vec3 spherePos, float sphereRadius, out vec3 intersecS, out vec3 normalS, out float resBis)
{

    //Distance between the center of the sphere and the initial position of the ray
    vec3 diff = rayPos - spherePos;
    
    //Operations with dot lead to the following equation:
    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(diff  , rayDir);
    float c = dot(diff  , diff  ) - sphereRadius * sphereRadius;
    
    //Computing of the discriminant to find the 2 slutions t1 and t2 (points moving in time hence each position p of a point
    //corresponds to rayPos(bse position) + t*rayDir with t moving
    float dist = b*b - 4.0*a*c;
    
    if (dist >= 0.0){
        float sdi = sqrt(dist);
        float den = 1. / (2.0 * a);
        float t1  = (-b - sdi) * den;
        float t2  = (-b + sdi) * den;
        
        bool res = (t1 > 0.0) || (t2 > 0.0);

        if(t1 > 0.0){
            intersecS = rayPos + t1*rayDir;
            normalS = normalize(intersecS - spherePos);
            resBis = t2;
            return t1;
        }
        else if (t2 > 0.0){
        	intersecS = rayPos + t2*rayDir;
            normalS = normalize(intersecS - spherePos);
            resBis = t1;
            return t2;
        }
     
    }  
    return -1.0;
}

float getPositiveMin(float val1, float val2){
	float res = min(val1, val2);
    if (res == -1.0)
        res = max(val1, val2);
    return res;
}


float rayCSG(vec3 rayPos, vec3 rayDir, out vec3 intersecPt,out vec3 normal, out int subObjectId) 
{   
    vec3 norm1, norm2, norm3, norm4 = normal;
    vec3 inter1,inter2, inter3, inter4 = intersecPt;
    float resBis1,resBis2, resBis3, resBis4 = 0.0; 
    float res1 = raySphere2(rayPos, rayDir, csgspherePos1, csgsphereRadius1, inter1, norm1, resBis1);
    float res2 = raySphere2(rayPos, rayDir, csgspherePos2, csgsphereRadius2, inter2, norm2, resBis2);
    float res3 = raySphere2(rayPos, rayDir, csgspherePos3, csgsphereRadius3, inter3, norm3, resBis3);
    float res4 = raySphere2(rayPos, rayDir, csgspherePos4, csgsphereRadius4, inter4, norm4, resBis4);
    float res = -1.0;

    float positiveMin = getPositiveMin(getPositiveMin(res1,res2),res3);
    
    
    //Launching test to see if we hit something
    if (positiveMin != -1.0){     
        //Test if we are in the intersection
        if((res2<=resBis1 && res1 <= res2) || (res1 <= resBis2 && res2 <= res1 )){
            //retrieving the first wall we hit in the intersection
            res = getPositiveMin(max(res1,res2), getPositiveMin(resBis1, resBis2));
            //If sphere3 is closer than intersection, it is it that we touch first
            if (getPositiveMin(res, res3) == res3)
                res = res3;
            //Test to see if sphere to cut (geometric cut) is in the front, and if it goes inside the current object
            if (getPositiveMin(res4, res) == res4){
                if(resBis4 < resBis3 && resBis4 >= res3 && res4 <= res3){
                    res = max(resBis4, resBis3);
                    if (res == resBis4)
                        res = -1.0;
                }
                else{
                    if(resBis4 < resBis2 && resBis4 >= res2 && res4 <= res2){
                        res = getPositiveMin(resBis4,resBis1);
                        if (res == resBis1)
                            res = -1.0;
                    }
                    if(resBis4 < resBis1 && resBis4 >= res1 && res4 <= res1){
                        res = getPositiveMin(resBis4,resBis2);
                        if (res == resBis2)
                            res = -1.0;
                    }
                }
            }

        }
        //If we are not in the intersection, seeing if we hit sphere3
        else if(res3 != -1.0)
            res = res3;
        // testing if hitting sphere3 first
        if (positiveMin == res3)
            res = res3;
        if (getPositiveMin(res4, res3) == res4){
            if(resBis4 < resBis3 && resBis4 >= res3 && res4 <= res3){
                res = getPositiveMin(resBis4, resBis3);
                if (res == resBis3)
                    res = -1.0;
            }
            else{
                    if(resBis4 < resBis2 && resBis4 >= res2 && res4 <= res2){
                        res = getPositiveMin(resBis4,resBis1);
                        if (res == resBis1)
                            res = -1.0;
                    }
                    if(resBis4 < resBis1 && resBis4 >= res1 && res4 <= res1){
                        res = getPositiveMin(resBis4,resBis2);
                        if (res == resBis2)
                            res = -1.0;
                    }
                }
        }
   	}
    



    if (res != -1.0){    
        if( res == res1 || res == resBis1){
            subObjectId = 1;
            normal = norm1;
            intersecPt = inter1;
        }
        else if( res == res2 || res == resBis2){
            subObjectId = 2;
            normal = norm2;
            intersecPt = inter2;
        }
        else if( res == res3 || res == resBis3){
            subObjectId = 3;
            normal = norm3;
            intersecPt = inter3;
        }
        else{
           subObjectId = 4;
           intersecPt = rayPos + res*rayDir;
           normal = -normalize(intersecPt - csgspherePos4);
            
        } 
    }
    return res;
}

int subObjectIdC = 0;

// The aim of this routine is to find the nearest intersection the ray has with ALL the objects
float computeNearestIntersection(vec3 rayPos, vec3 rayDir,
                                 out int objectId, out vec3 intersecI, out vec3 normalI)
{
    // Set the default value when no intersection is found: we hit the 'sky'
    float minDist  = FLT_MAX;
    objectId = skyId;
    
    // Test the sphere
    vec3 intersecS, normalS;
    float distS = raySphere(rayPos, rayDir, spherePos, sphereRadius, intersecS, normalS);
    if ((distS > 0.0) && (distS < minDist))
    {
        objectId  =  sphereId;
        minDist   =     distS;
        intersecI = intersecS;
          normalI =   normalS;
    }
    
    
    //Test the CSG object
    vec3 intersecC, normalC;
    float distC = rayCSG(rayPos, rayDir, intersecC, normalC, subObjectIdC);
    if ((distC > 0.0) && (distC < minDist))
    {
        objectId  =   csgId;
        minDist   =     distC;
	    intersecI = intersecC;
    	  normalI =   normalC;
    }
    
    
    
    
    // Test the plane
    vec3 intersecP, normalP;
    float distP =  rayPlane(rayPos, rayDir,  planePos,  planeNormal, intersecP, normalP);
    if ((distP > 0.0) && (distP < minDist))
    {
        objectId  =   planeId;
        minDist   =     distP;
	    intersecI = intersecP;
    	  normalI =   normalP;
    }
    
    // To remain coherent with the raySphere & rayPlane function that returns -1 when no
    // intersetion is found, we add the following two lines:
    if (objectId == skyId)
        minDist = -1.0;
    
    return minDist;
}


vec3 getSphereColorAtPoint(vec3 pt)
{
    return sphereCol;
}

//----------------------------------------------------------------------------------------------


vec3 getCSGColorAtPoint(vec3 pt){
    if (subObjectIdC == 1)
        return csgCol1;
    if (subObjectIdC == 2)
        return csgCol2;
    if (subObjectIdC == 3)
        return csgCol3;
    if (subObjectIdC == 4)
        return csgCol4;
    return vec3(3);
}



// pt is assumed to be on the plane surface
vec3 getPlaneColorAtPoint(vec3 pt)
{
    vec3 worldX = vec3(1,0,0);
    vec3 axisX  = normalize(worldX - dot(worldX, planeNormal) * planeNormal);
    
    // We then find the plane Y-axis thanks to the cross-product
    // properties with orthonormal basis
    vec3 axisY  = normalize(cross(planeNormal, axisX));

    // Now, find the coordinate of the input point according to this texture coordinate frame
    vec3 diff = pt - planePos; 
    float u = dot(diff, axisX);
    float v = dot(diff, axisY);
    
    // Finally, apply the checkboard pattern by using this very concise formula:
    return mod(floor(u * 0.5) + floor(v * 0.5), 2.0) < 1.0  ? planeCol1 : planeCol2;
}


vec3 getObjectColorAtPoint(int objectId, vec3 pt, out Material objectMat)
{
    if (objectId == sphereId)
    {
        objectMat = sphereMat;
        return getSphereColorAtPoint(pt);
    }
    else if (objectId == planeId)
    {
        objectMat = planeMat;
        return getPlaneColorAtPoint(pt);
    }
    else if (objectId == csgId)
    {
        objectMat = csgMat;
        return getCSGColorAtPoint(pt);
    }
    return skyCl;
}


float getShadowFactorAtPoint(vec3 intersec, vec3 reflec, vec3 normal, Material objectMat,LightInfo light){
    
    int objectId2;
    vec3 intersec2, normal2; 
    intersec += 0.1 * normal;
    float d = computeNearestIntersection(intersec, light.pos - intersec, objectId2, intersec2, normal2);
    if (d < 0.0){
    	return 1.0;
    }
    else{
    	return 0.2;
    }
}


void computeCameraRayFromPixel( in vec2 pixCoord, out vec3 rayPos, out vec3 rayDir){

    //focal distance, trigonometric computing
    float f = 1.0 / tan(radians(cameraFovY) / 2.0);
    //ez
    vec3 c_z = normalize(cameraTarget - cameraPos);
    //Defining a unitary vector, inveted from cy perfect
    vec3 c_y2 = vec3(0,1,0);
    //vector product to find c_x
    vec3 c_x = normalize(cross(- c_y2, c_z));
    //pareil
    vec3 c_y = normalize(cross(c_z, c_x));
    
    //Recadring the coordinate system
    vec2 pt = (2.*pixCoord - iResolution.xy) / iResolution.y;

    //postiion of the ray, issued from the camera
    rayPos = cameraPos;
    
    //direction of the ray, coordx * computed vectors except c_y2
    rayDir = normalize(pt.x * c_x - pt.y * c_y + f * c_z);
}



vec3 computePhongShading(vec3 objectCol, Material objectMat, vec3 normalS, vec3 L, vec3 R, vec3 V, float shadowFactor, vec3 lightCol){

    vec3 ambiant = objectMat.Ka * ambiantCol;
    vec3 diffuse = objectMat.Kd * objectCol * lightCol * max(dot(normalS,L), 0.0) * shadowFactor;
    vec3 specular = objectMat.Ks * lightCol * pow(max(dot(R,V), 0.0), objectMat.Kn) * shadowFactor;
    
    vec3 phongCol = ambiant + diffuse + specular;
    return phongCol;
    
}



vec3 RaytraceAtPixelCoord(vec2 pixCoord)
{
    vec3 rayPos, rayDir;
    vec3 resCol;
    computeCameraRayFromPixel(pixCoord, rayPos, rayDir);
    
    ShadeInfo infos[MAX_NB_BOUNCES];
    int nbBounces = 0;
    do
    {
        // Test ray-objects intersections and find the nearest one
        int objectId;
        vec3 intersec, normal;
        float dist = computeNearestIntersection(rayPos, rayDir, objectId, intersec, normal);
        
        // We did not hit the sphere, so we have the sky color
        if (dist <= 0.0)
        {
            resCol=skyCl;
            infos[nbBounces].shadedCol = skyCl;
            infos[nbBounces].Ks = 1.0;
            break;
        }
        
        else{
            vec3 Cam;
            for (int l = 0; l < NB_LIGHTS ; l++){
                
                // unit-vector going from the surface toward the light
                vec3 L = normalize(lights[l].pos - intersec);

                Cam = normalize(cameraPos - intersec);

                //unit-vector of the reflection direction of the light at the surface point
                vec3 R = 2.0*dot(normal, L)*normal-L;

                //unit-vector going from the surface point toward the camera
                vec3 V = -rayDir;

                Material objectMat;

                vec3 objectCol = getObjectColorAtPoint(objectId, intersec, objectMat);
                
                //get the color of the object for motion blur
                if(nbBounces == 0)
                    objectColForBlurr = objectCol;

                shadowFactor = getShadowFactorAtPoint(intersec, L, normal, objectMat, lights[l]);

                //Apply the Phong shading to compute the color
                //of the surface point as seen from the camera

                //fragColor = vec4(computePhongShading(objectCol, objectMat, normal, L, R, V, shadowFactor), 1);

                 // Store the information we gathered for that surface point
                infos[nbBounces].shadedCol += computePhongShading(objectCol, objectMat, normal, L, R, V, shadowFactor, lights[l].col);
                infos[nbBounces].Ks += objectMat.Ks;  

                
            }
            
            infos[nbBounces].shadedCol = infos[nbBounces].shadedCol / 2.0;
            infos[nbBounces].Ks = infos[nbBounces].Ks / 2.0;
            // Bounce from the surface towards the reflected direction of the ray
                rayPos = intersec + 0.001  * normal;
                rayDir = 2.0 * dot(normal, Cam) * normal - Cam;
        
        }
              
        nbBounces++;
    }
    while (nbBounces < MAX_NB_BOUNCES);
    
    
    for (int i = 0; i < nbBounces; i+=1){
    	resCol = infos[nbBounces - i - 1].shadedCol + infos[nbBounces - i - 1].Ks* resCol;
    }
    
    return resCol;
}

vec2 noisePix(vec2 location, vec2 delta){
    const vec2 c = vec2(12.9898, 78.233);
    const float m = 43758.5453;
    return vec2(
        fract(sin(dot(location +	  delta			   , c)) * m),
        fract(sin(dot(location + vec2(delta.y, delta.x), c)) * m)
    );
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    lights[0].pos = lightPos1;
    lights[0].col = lightCol1;
    lights[0].power = lightPow1;
    lights[1].pos = lightPos2;
    lights[1].col = lightCol2;
    lights[1].power = lightPow2;
    AnimateScene(iTime);
  // Compute the ray to be casted through the pixel towards the 3D scene
    
    //Anti-aliasing
    //Non-Stochastic approach
    
    vec3 res;
    
    if (!vanilla){
    
        vec2 subPixels[PIXEL_SAMPLING_GRID_SIZE];
        
        if (!stoch){
        subPixels[0] = vec2(0.25, 0.20) + fragCoord;
        subPixels[1] = vec2(0.25, 0.40) + fragCoord;
        subPixels[2] = vec2(0.25, 0.60) + fragCoord;
        subPixels[3] = vec2(0.25, 0.80) + fragCoord;
        subPixels[4] = vec2(0.75, 0.20) + fragCoord;
        subPixels[5] = vec2(0.75, 0.40) + fragCoord;
        subPixels[6] = vec2(0.75, 0.60) + fragCoord;
        subPixels[7] = vec2(0.75, 0.80) + fragCoord;
        }
        
        for (int i=0; i<PIXEL_SAMPLING_GRID_SIZE; i+=1){
            if (blurr)
                subPixels[i] += vec2(i, i);
            if (motionBlurr){
                if (objectColForBlurr != planeCol1 && objectColForBlurr != planeCol2) //Astuce pas du tout durable et pertinente dans un vrai cas, mais ça marche
                   subPixels[i] += vec2(i, i);
            }
            if (stoch)
            	res += RaytraceAtPixelCoord(fragCoord + noisePix(fragCoord, vec2(5.0,5.0))); //stochastic, works weird
            else
            	res += RaytraceAtPixelCoord(subPixels[i]);
        }
        
	
        res = res/float(PIXEL_SAMPLING_GRID_SIZE);
        
        
        //fail for the motion blurr ^_^ False theory : we get an interpolation of the buffer's images which in the end
        //creates a coherent image
        /*
        if (motionBlurr){
            if (!firstLoop){
                for (int j=1; j<NB_MOTIONS_BUFFER; j+=1){
                    imgBuffer[j] = imgBuffer[j-1];
                	res += imgBuffer[j];
                }
                imgBuffer[0] = res;
            }
            if(firstLoop){
                for (int j=0; j<NB_MOTIONS_BUFFER; j+=1){
                    imgBuffer[j] = res;
                }
                res = res * float(NB_MOTIONS_BUFFER);
                firstLoop = false;
            }
            res = res/float(NB_MOTIONS_BUFFER);
        }
		*/        

    }
    else{
        res += RaytraceAtPixelCoord(fragCoord);
    }
    fragColor = vec4(res, 1);
    
    
}
	    
    
    
    
    
