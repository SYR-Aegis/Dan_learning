#include <iostream>
#include <cstring>
using namespace std;

char& find(char a[], char c, bool& success) { //a 바꾸면 원본도 바뀜
	char* ptr = strchr(a, c);

	if (ptr != NULL) {
		success = true;
		return *(ptr);
	}
	success = false;

}

int main() {
	char s[] = "Mike";
	bool b = false;
	char& loc = find(s, 'M', b); //값에 의한 호출 같지만 s는 배열이라 메모리공유
	if (b == false) {
		cout << "M을 발견할 수 없다." << endl;
		return 0;
	}
	loc = 'm'; //M위치에 m기록
	cout << s << endl;
}