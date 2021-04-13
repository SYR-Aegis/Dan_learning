
#include <stdio.h>
#include <net/ethernet.h>
#include <netinet/ip.h>
#include <netinet/ip_icmp.h>
#include <ctype.h>
#include <netinet/tcp.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct _Packet{
    u_char* packet_stream;
    u_int packet_size;
}Packet;

typedef struct _GlobalHeader{
    uint32_t magic_number; 
    uint16_t version_major;
    uint16_t version_minor; 
    int32_t thiszone; 
    uint32_t sigfigs; 
    uint32_t snaplen; 
    uint32_t network;
}GlobalHeader;

typedef struct _Pcap{
    uint32_t ts_sec; 
    uint32_t ts_usec; 
    uint32_t incl_len; 
    uint32_t orig_len;
}Pcap;

/*
ip_len =  ip_header_size(20 Byte) + tcp_header_size(20 Byte) + tcp_option_size + tcp_payload_size 
tcp_data_offset = tcp_data_offset * 4
tcp_data_len = ip_len - (tcp_data_offset + ip_header_size) * 4 
tcp_option_len = (tcp_data_offset * 4) - tcp_header_size
*/

void analysis_packet(Packet* packet);
Packet* get_packet_byte(const char* filename);
void error_handling(const char* msg);
void print_tcp(struct tcphdr* p_tcphd, u_short ip_total_len, u_int ip_header_len);
void print_tcp_payload(u_char* payload, u_int len);

int main(int argc, char** argv){
    if(argc != 2){
        printf("Usage : %s with one Argument\n", argv[1]);
        exit(1);
    }

    Packet* packet = get_packet_byte(argv[1]);
    analysis_packet(packet);
    free(packet->packet_stream);
    free(packet); 
}

Packet* get_packet_byte(const char* filename){
    Packet* packet = (Packet*)malloc(sizeof(Packet));
    int fd = open(filename, O_RDONLY);
    if(fd <= 0){
        error_handling("OPEN ERROR");
    }
    int f_size = lseek(fd, 0, SEEK_END);
    u_char* packet_stream = (u_char*)malloc(f_size + 1);  
    lseek(fd, 0, SEEK_SET); 
    read(fd, packet_stream, f_size);
    packet->packet_size = f_size; 
    packet->packet_stream = packet_stream;
    return packet;
}

//htons or htonl 중요 Network byte to host byte 

void analysis_packet(Packet* packet){
    struct ether_header *p_ethr;
    struct ip *p_iphdr;
    struct tcphdr *p_tcphdr;
    int ether_size = sizeof(struct ether_header);
    int ip_size = sizeof(struct ip); 
    int tcp_size = sizeof(struct tcphdr);

    u_int32_t p_size = sizeof(GlobalHeader) + sizeof(Pcap); //현재 패킷 Pointer 사이즈
    //모두다 TCP Packet임
    for(;p_size<=packet->packet_size;){
        p_ethr = (struct ether_header*)(packet->packet_stream + p_size);
        p_size += ether_size;
        p_iphdr = (struct ip*)(packet->packet_stream + p_size);
        u_short remain_size = htons(p_iphdr->ip_len) - (ip_size); //ip header와 tcp header의 사이즈를 total ip len에서 뺌 나중에 이 값을 더해서 다음 packet 오프셋으로 찾음
        p_size += ip_size;
        p_tcphdr = (struct tcphdr*)(packet->packet_stream + p_size);
        print_tcp(p_tcphdr, htons(p_iphdr->ip_len), p_iphdr->ip_hl);
        p_size += remain_size + sizeof(Pcap); //Why added sizeof(Pcap) == https://wiki.wireshark.org/Development/LibpcapFileFormat
    }
    return;
}

void print_tcp(struct tcphdr* p_tcphd, u_short ip_total_len, u_int ip_header_len){
    //https://stackoverflow.com/questions/6639799/calculate-size-and-start-of-tcp-packet-data-excluding-header 
    printf("Source Port and Dest Port : %d %d\n",htons(p_tcphd->source), htons(p_tcphd->dest));
    u_int payload_len = ip_total_len - ((p_tcphd->th_off + ip_header_len) * 4); // TCP  Data 위치
    if(payload_len > 0){
        printf("DATA! : "); //p_tcphd->th_off * 4 - sizeof(struct tcphdr) 의 크기는 Option의 크기 
        print_tcp_payload((u_char*)((u_char*)p_tcphd + (p_tcphd->th_off * 4)), payload_len);
    }
}

void print_tcp_payload(u_char* payload, u_int len){
    for(int i=0;i<len;i++){
        printf("%c", payload[i]);
    }
    printf("\n");
}

void error_handling(const char* msg){
    printf("Error:%s\n", msg);
    exit(1);
}