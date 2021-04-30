#include <iostream>
#include <string>
#include <cstring>
using namespace std;

class Person {
	string name;
	string tel;
public:
	Person() {};
	string getName() {
		return name;
	}
	string getTel() {
		return tel;
	}
	void set(string name, string tel) {
		this -> name = name;
		this->tel = tel;
	}
};

int main() {
	Person p_Array[3];
	string name;
	string tel;
	
	cout << "이름과 전화번호를 입력해주세요"<<endl<<"사람 1>> ";
	cin >> name >> tel;
	p_Array[0].set(name, tel);
	cout <<  "사람 2>> ";
	cin >> name >> tel;
	p_Array[1].set(name, tel);
	cout << "사람 3>> ";
	cin >> name >> tel;
	p_Array[2].set(name, tel);

	cout << "모든 사람의 이름은 ";
	for (int i = 0; i < 3; i++) {
		cout << p_Array[i].getName() << " ";
	}
	cout << "전화번호를 검색합니다. 이름을 입력하세요>>";
	cin >> name;
	for (int i = 0; i < 3; i++) {
		if (p_Array[i].getName()==name) {
			cout <<endl<<"전화 번호는 "<< p_Array[i].getTel();
		}
	}
}