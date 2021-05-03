#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/eventfd.h>
#include <stdlib.h>
#include <sys/epoll.h>

void* io_wait_thread(void* arg);
void* worker_thread(void* arg); 


pthread_mutex_t mutx; 

int efd = 0;
const int thread_num = 5; 

int main(void){
    pthread_t pthread[thread_num];
    pthread_t io_pthread;
    uint64_t arg_data[thread_num];

    printf("EFD_NONBLOCK:%d\n",EFD_NONBLOCK);
    efd = eventfd(0, EFD_NONBLOCK);
    if(efd < 0){
        perror("eventfd()"); 
        exit(1);
    }
    pthread_mutex_init(&mutx, NULL);
    int thr_id = pthread_create(&io_pthread, NULL, io_wait_thread, NULL);
    if(thr_id < 0){
        perror("pthread_create()"); 
        exit(1); 
    }
    for(uint64_t i=0;i<thread_num;i++){
        arg_data[i] = i+1;
        thr_id = pthread_create(&pthread[i], NULL, worker_thread, (void*)&arg_data[i]);
        if(thr_id < 0){
            perror("pthread_create()"); 
            exit(1); 
        }
    } 

    pthread_join(io_pthread, NULL); 
    for(int i=0;i<thread_num;i++){
        pthread_join(pthread[i], NULL); 
    }
}

void* io_wait_thread(void* arg){
    int ep_fd = 0; 
    uint64_t signal_number = 0;
    struct epoll_event events[10];
    //struct epoll_event* events = (struct epoll_event*)malloc(sizeof(struct epoll_event) * thread_num); 
    ep_fd =  epoll_create1(EPOLL_CLOEXEC);
    if(ep_fd < 0){
        perror("epoll_create1()\n");
        exit(1);
    }
    struct epoll_event want_trace_event;
    want_trace_event.events = EPOLLHUP | EPOLLERR | EPOLLIN;
    want_trace_event.data.fd = efd; 
    int ret = epoll_ctl(ep_fd, EPOLL_CTL_ADD, efd, &want_trace_event);
    if(ret < 0){
        perror("epoll_ctl()");
        exit(1);
    }
    while(1){
        ret = epoll_wait(ep_fd, &events[0], thread_num, 5000);
        if(ret < 0){
            perror("epoll_wait()");
            exit(1);
        }
        else if(ret == 0){
            printf("TIMEOUT!\n");
            exit(1);
        }
        for(int i=0;i<ret;i++){
            if(events[i].events & EPOLLHUP){
                perror("epoll eventfd has epoll hup\n"); 
                exit(1);
            }
            else if(events[i].events & EPOLLERR){
                perror("epoll eventfd has epoll error\n"); 
                exit(1);
            }
            else if(events[i].events & EPOLLIN){
                int event_fd = events[i].data.fd; 
                
                ret = read(event_fd, &signal_number, sizeof(signal_number));
                
                if(ret < 0){
                    perror("read()"); 
                    exit(1);
                }
                printf("get signal data %ld\n", signal_number);
            }
        }
    }
    
}

void* worker_thread(void* arg){
    uint64_t signal_number = *((uint64_t*)arg);
    int ret = write(efd, &signal_number, sizeof(signal_number));
    if(ret < 0){
        perror("write()");
        exit(1);
    }
    printf("thread number %lx send Signal Number %ld!\n", pthread_self(), signal_number);
}