#include <iostream>
#include <cstring>

using namespace std;

int main() {
	char name[100];
	int max_size;
	char max_name[30];

	cout << "5 명의 이름을 ';'으로 구분하여 입력하세요" << endl << ">>";

	for (int i = 1; i <= 5;i++) {
		cin.getline(name, 100, ';'); //입력 받음
		cout << i << " : " << name << endl;

		if (i == 1) {
			max_size = strlen(name);
			strcpy_s(max_name,name);
		}
		if (max_size < strlen(name)) {
			strcpy_s(max_name, name);
			max_size = strlen(name);
		}
	}//기준이 ;이고 엔터 치면 싹 다 출력됨..ㄷㄷ//getline 아주 구석구석 공부필요

	cout << "가장 긴 이름은" << max_name;
}

