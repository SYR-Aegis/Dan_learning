#include <iostream>
using namespace std;

bool bigger(int a, int b, int& big) {
	if (a == b) {
		return true;
	}
	else {
		if (a > b) {
			big = a;
		}
		else {
			big = b;
		}
		return false;
	}
}

int main() {
	int big;
	int a, b;
	cout << "정수 두개를 입력하시오." << endl;
	cin >> a >> b;
	bool result = bigger(a, b,big);
	if (result == true) {
		cout << "정수 두개가 같습니다." << endl;
	}
	else {
		cout << "더 큰 수는"<<big<<"입니다" << endl;
	}

}