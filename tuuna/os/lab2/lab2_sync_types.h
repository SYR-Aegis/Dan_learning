/*
*   DKU Operating System Lab
*           Lab2 (Vehicle Production Problem)
*           Student id : 
*           Student name : 
*
*   lab1_sync_types.h :
*       - lab2 header file.
*       - must contains Vehicle Production Problem's declations.
*
*/

#ifndef _LAB2_HEADER_H
#define _LAB2_HEADER_H

#include <stdio.h>
#include <fcntl.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <pthread.h>
#include <asm/unistd.h>
#include <stdlib.h>

#define MAX_SIZE 1000000

/*
 * You need to Declare functions in  here
 */

typedef struct _Node {
	int car_num;
	struct _Node *next;
}Node;

typedef struct _CarQueue {
	pthread_mutex_t mutx;
	pthread_cond_t consumer; 
	pthread_cond_t producer;
	int balance; //현재 큐 길이
	Node *head;
	Node *tail;
	int consumed_car_numof;
	int produced_car_numof;
}CarQueue;

typedef struct _Consumer{
	CarQueue* car_queue; 
	int car_num; 
}Consumer; 

typedef struct _Producter{
	CarQueue* car_queue; 
	int car_num;
}Producer;

void Handler(int total_car, int time_slice);
void* produce_car_thread(void* arg);
void* consume_car_thread(void* arg);
int get_car(CarQueue* car_queue, int car_num);
int insert_car(CarQueue* car_queue, int car_num);

#endif /* LAB2_HEADER_H*/


