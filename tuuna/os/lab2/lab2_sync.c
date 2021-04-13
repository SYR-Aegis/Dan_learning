/*
*   DKU Operating System Lab
*           Lab2 (Vehicle production Problem)
*           Student id : 
*           Student name : 
*
*   lab2_sync.c :
*       - lab2 main file.
*       - must contains Vehicle production Problem function's declations.
*
*/

#include <aio.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <pthread.h>
#include <asm/unistd.h>

#include "include/lab2_sync_types.h"

/*
 * you need to implement Vehicle production Problem. 
 */

const int false = 0;
const int true = 1;

const int size = 10;

void Handler(int total_car, int time_slice){

    struct timeval startTime, endTime; 
    double diffTime; 
    pthread_t consumer_thread[size];
    pthread_t producer_thread[size];
    CarQueue* car_queue = (CarQueue*)malloc(sizeof(CarQueue));
    car_queue->balance = 0; 
    car_queue->head = NULL;
    car_queue->tail = NULL;
    car_queue->consumed_car_numof = 0;
    car_queue->produced_car_numof = 0;
    pthread_mutex_init(&car_queue->mutx, NULL);

    gettimeofday(&startTime, NULL);
    for(int i=0;i<size;i++){
        Producer* producer = (Producer*)malloc(sizeof(Producer));
        producer->car_queue = car_queue; 
        producer->car_num = i+1;

        Consumer* consumer = (Consumer*)malloc(sizeof(Consumer));
        consumer->car_queue = car_queue; 
        consumer->car_num = i+1;
        pthread_create(&producer_thread[i], NULL, consume_car_thread, (void*)consumer);
        pthread_create(&consumer_thread[i], NULL, produce_car_thread, (void*)producer);
    }

    for(int i=0;i<size;i++){
        pthread_join(producer_thread[i], NULL);
        pthread_join(consumer_thread[i], NULL);
    }
    printf("Produce Car %d\n", car_queue->produced_car_numof);
    printf("Consume Car : %d\n", car_queue->consumed_car_numof);
    printf("Car Size : %d\n", car_queue->produced_car_numof);
    free(car_queue);
    gettimeofday(&endTime, NULL); 
    diffTime = ( endTime.tv_sec - startTime.tv_sec ) + (( endTime.tv_usec - startTime.tv_usec ) / 1000000);
    printf("Time : %f\n", diffTime);
    return;
}

void* produce_car_thread(void* arg){
    Producer* producer = (Producer*)arg;
    CarQueue* car_queue = producer->car_queue; 
    int car_num = producer->car_num;
    //printf("Producer THread : %d\n", car_num);
    int fin_flag;
    int current_size = 0; 
    int max_size = MAX_SIZE /size;
    while(1){
        fin_flag = insert_car(car_queue, car_num);
        current_size+=1;
        if(current_size == max_size){
            //printf("%d Producer Thread Exited\n", car_num);
            break;
        }
    }
    
    free(producer);
}

void* consume_car_thread(void* arg){
    Consumer* consumer = (Consumer*)arg;
    CarQueue* car_queue = consumer->car_queue; 
    int car_num = consumer->car_num;
    int current_size = 0;
    int max_size = MAX_SIZE / size;
    //printf("Consumer THread : %d\n", car_num);
    int get_flag;
    while(1){
        get_flag = get_car(car_queue, car_num);
        if(get_flag){
            current_size += 1;
        }
        if(current_size == max_size){
            //printf("%d Consumer Thread Exited\n", car_num);
            break;
        }
    }
    
    free(consumer);
}

//큐에서 카 하나를 뺌 이때, 요청된 차량이 있어야 함
//원하는게 없을 때는?  지금은 없으면 다시 get_car 함수 호출하는 걸로 후순위로 넘김
//소비자는 원하는 카가 없을 때 무한정 기다려야 하는가 (생산량 제한 각 각 총 생산의 /5 씩 )
int get_car(CarQueue* car_queue, int car_num){
    int get_flag = false;
    int max_size = MAX_SIZE / size;
    int current_size = 0;
    pthread_mutex_lock(&car_queue->mutx);
    while(car_queue->balance == 0){
        //printf("OH! Empty Balance %d Thread will wait\n", car_num);
        pthread_cond_wait(&car_queue->producer, &car_queue->mutx);
    }
    Node* curr = car_queue->head; 
    Node* prev = curr;
    while(curr){
        if(curr->car_num == car_num){
            get_flag = true;
            //printf("Consumer Get Car : %d\n", curr->car_num);
            if(curr == car_queue->head){
                car_queue->head = curr->next;
            }
            else if(curr == car_queue->tail){
                prev->next = NULL; 
                car_queue->tail = prev;
            }else{
                prev->next = curr->next; 
            }
            car_queue->consumed_car_numof +=1;
            current_size += 1;
            car_queue->balance -= 1; 
            pthread_cond_signal(&car_queue->consumer);
            free(curr);
            break;
        }
        prev = curr;
        curr = curr->next; 
    }
    pthread_mutex_unlock(&car_queue->mutx);
    return get_flag;
}

//큐에 카를 넣음
int insert_car(CarQueue* car_queue, int car_num){
    int fin_flag = false;
    Node* new_node = (Node*)malloc(sizeof(Node));
    new_node->car_num = car_num;
    new_node->next = NULL;
    pthread_mutex_lock(&car_queue->mutx);
    while(car_queue->balance == 5){
        //printf("OH Exceed Balanace! %d Thread will wait\n", car_num);
        pthread_cond_wait(&car_queue->consumer, &car_queue->mutx);
    }
    //->
    //printf("Producer Insert Car %d\n", new_node->car_num);
    if(car_queue->head == NULL){
        car_queue->head = new_node; 
        car_queue->tail = new_node; 

    }else{
        car_queue->tail->next = new_node; 
        car_queue->tail = new_node;
    }
    car_queue->balance += 1;
   
    car_queue->produced_car_numof += 1;
    pthread_cond_signal(&car_queue->producer);
    pthread_mutex_unlock(&car_queue->mutx);
    return fin_flag;
}


void lab2_sync_usage(char *cmd) {
	printf("\n Usage for %s : \n",cmd);
    printf("    -c: Total number of vehicles produced, must be bigger than 0 ( e.g. 100 )\n");
    printf("    -q: RoundRobin Time Quantum, must be bigger than 0 ( e.g. 1, 4 ) \n");
}

void lab2_sync_example(char *cmd) {
	printf("\n Example : \n");
    printf("    #sudo %s -c=100 -q=1 \n", cmd);
    printf("    #sudo %s -c=10000 -q=4 \n", cmd);
}

int main(int argc, char* argv[]) {
	char op;
	int n; char junk;
	int time_quantum = 0; //타임 슬라이스
	int total_car = 0; //생산할 차량


	if (argc <= 1) {
		lab2_sync_usage(argv[0]);
		lab2_sync_example(argv[0]);
		exit(0);
	}

	for (int i = 1; i < argc; i++) {
		if (sscanf(argv[i], "-c=%d%c", &n, &junk) == 1) {
			total_car = n;
		}
		else if (sscanf(argv[i], "-q=%d%c", &n, &junk) == 1) {
			time_quantum = n;
		}
		else {
			lab2_sync_usage(argv[0]);
			lab2_sync_example(argv[0]);
			exit(0);
		}
	}	
	Handler(total_car, time_quantum);
	return 0;
}
