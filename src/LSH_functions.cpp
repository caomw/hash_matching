// LSH (Locality Sensitive hashing) functions 
// see reference Locality-Sensitive Hashing Scheme Based on p-Stable Distributions 
// (by Alexandr Andoni, Mayur Datar, Nicole Immorlica, Piotr Indyk, and Vahab Mirrokni), 
// appearing in the book Nearest Neighbor Methods in Learning and Vision: Theory and Practice, 
// by T. Darrell and P. Indyk and G. Shakhnarovich (eds.), MIT Press, 2006.

#include "LSH_functions.h" 

// Computes (a.b)mod UH_PRIME_DEFAULT. b is coded as <nBPieces> blocks
// of size totaling <size>. <a> is of length <size>.
// inline Uns32T computeBlockProductModDefaultPrime(vector<uint> random_vector, vector<uint> bucket_function, IntT nBPieces, IntT size){
//   LongUns64T h = 0;
//   IntT j = 0;
//   IntT bPiece = 0;
//   IntT bPieceSize = size / nBPieces;
//   IntT i = 0;
//   for(IntT bPiece = 0; bPiece < nBPieces; bPiece++){
//     for(IntT j = 0; j < bPieceSize; j++, i++){
//       h = h + (LongUns64T)a[i] * (LongUns64T)b[bPiece][j];
//       h = (h & TWO_TO_32_MINUS_1) + 5 * (h >> 32);
//       if (h >= UH_PRIME_DEFAULT) {
// 	h = h - UH_PRIME_DEFAULT;
//       }
//       CR_ASSERT(h < UH_PRIME_DEFAULT);
//     }
//   }
//   return h;
// }

// Computes (a.b)mod UH_PRIME_DEFAULT.

Uns32T computeProductModDefaultPrime(vector<float> random_vector, vector<double> bucket, int size){
  LongUns64T h = 0; // h has to be int since it is the index in a table
  int i = 0;
  for(i = 0; i < size; i++){
    h = h + (LongUns64T)(random_vector[i] * bucket[i]); // sum ri*ai
    h = (h & TWO_TO_32_MINUS_1) + 5 * (h >> 32);  // low(ri*ai)+5*high(ri*ai)
    if (h >= UH_PRIME_DEFAULT) { 
      h = h - UH_PRIME_DEFAULT;
    }
    ROS_ASSERT(h < UH_PRIME_DEFAULT); // integrity check 
  }
  return h;
}

// Compute fuction ((rndVector . data)mod prime)mod hashTableSize
// Vectors <random_vector> and <bucket> are assumed to have length <size>.
// size--> bucket size; prime--> prime number needed to compute the hash; 
//hashTablesize--> number of elements in the hash table = equal to the number of keyimages

Uns32T computeUHashFunction(vector<float> random_vector, vector<double> bucket, IntT size, Uns32T prime, Int32T hashTableSize){
  ROS_ASSERT(prime == UH_PRIME_DEFAULT);
  ROS_ASSERT(random_vector != NULL);
  ROS_ASSERT(bucket != NULL);

  Uns32T h = computeProductModDefaultPrime(random_vector, bucket, size) % hashTableSize;

  ROS_ASSERT(h >= 0 && h < hashTableSize); // return the index in the hash table
 
  return h;
}

// Generate 'd' random numbers
vector<float> compute_random_vector(uint seed, int size)
{
  srand(seed);
  vector<float> h;
  h.clear();
  ROS_ASSERT(MIN_HASH_RND <= MAX_HASH_RND);
  for (int i=0; i<size; i++)
  {
    //float val = ((float(rand()) / float(RAND_MAX)) * (1 + 1)) -1.0; // generate the random values
    // float val = ( (float(rand()) / (float)(RAND_MAX)) * MAX_HASH_RND  ) ; // generate the random values between 1 and MAX_HASH_RND
    //  float val = ( ( (float)(rand()) % MAX_HASH_RND )+ 1); // generate the random values between 1 and MAX_HASH_RND
    // output = min + (rand() % (int)(max - min + 1))
    //  float val = ( ( (rand()) % (MAX_HASH_RND-1+1) )+ 1); // generate the random values between 1 and MAX_HASH_RND
    // get a random real distributed uniformly
    float val = (  ((MAX_HASH_RND-MIN_HASH_RND)*(float)(rand()))/(float)(RAND_MAX)  ) + MIN_HASH_RND; // generate the random values between 1 and MAX_HASH_RND
    ROS_ASSERT(val >= MIN_HASH_RND && r <= MAX_HASH_RND);
    ROS_INFO_STREAM("random data " << rand() << ";--" << MAX_HASH_RND-1+1 << ";" << (rand()) % (MAX_HASH_RND-1+1) );
    h.push_back(val); // storMIN_HASH_RNDhe random value in the vector
  }
  return h;
}