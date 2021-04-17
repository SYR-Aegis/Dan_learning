#include <iostream>
#include <cstring>
using namespace std;

int main() {
	int profit=0;
	char coffee[100];
	int num;
	int price;

	cout << "에스프레소 2000원, 아메리카노 2300원, 카푸치노 2500원"<<endl;
	while (profit < 20000) {
		cout << "주문>> ";
		cin >> coffee >> num;

		if (strcmp(coffee, "에스프레소") == 0) {
			price = num * 2000;
		}
		else if (strcmp(coffee, "아메리카노") == 0) {
			price = num * 2300;
		}
		else if (strcmp(coffee, "카푸치노") == 0) {
			price = num * 2500;
		}

		cout << price << "입니다. 맛있게 드세요" << endl;
		profit += price;
	}
	cout << "오늘" << price << "원을 판매하여 카페를 닫습니다. 내일 봐요~~~";
	return 0;
}