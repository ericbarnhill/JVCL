
__kernel void stockhamStride(  __read_only float * vec1, __read_only float * vec2, __global float * vecOut) 
{
	int block = get_group_id(0);
	int x = get_local_id(0);
	int index = block*blockSize+x;
}
