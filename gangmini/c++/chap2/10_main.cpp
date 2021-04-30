#include <iostream>
#include <cstring>
using namespace std;

int main() {
	char str[10];
	cout << "문자열 입력>>";
	cin >> str;

	for (int i = 0; i < strlen(str); i++) {
		for (int h = 0; h <= i; h++) {
			cout << str[h];
		}
		cout << endl;
	}
}