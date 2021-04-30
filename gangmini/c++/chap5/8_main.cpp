#include <iostream>
using namespace std;

class MyIntStack {
	int* p; //스택메모리로 사용할 포인터
	int size; //스택의 최대 크기
	int tos; // 스택의 탑을 가리키는 인덱스

public:
	MyIntStack();
	MyIntStack(int size);
	MyIntStack(MyIntStack& s); //복사생성자
	~MyIntStack();
	bool push(int n); //정수 n을 스택에 푸시//꽉 차면 flase
	bool pop(int& n);//스택의 탑 값을 n에 pop//비어있으면 false
};

MyIntStack::MyIntStack()
{	this->size = 10;
	this->p = new int[size];
	tos = 100;
}

MyIntStack::MyIntStack(int size)
{   this->size = size;
    this->p = new int[size];
	tos = 100;
}

MyIntStack::~MyIntStack() {
	delete p;
}

MyIntStack::MyIntStack(MyIntStack& s) {
	this->size = s.size;
	this->p = new int[this->size];
	if (tos != 100) {
		for (int i = 0; i <= s.tos; i++) {
			this->p[i] = s.p[i];
		}
	}
	this->tos = s.tos;

}

bool MyIntStack::push(int n) {
	if (tos == 100) {
		tos = 0;
		p[tos] = n;
		return true;
	}
	if (tos != 10) {
		tos++;
		p[tos] = n;
		return true;
	}
	return false;
}

bool MyIntStack::pop(int& n) {//pop하면 top값 없애나?
	if (*p != NULL) {
		n = p[tos];
		p[tos] = NULL;
		if (tos == 0) {
			tos = 100;
			return true;
		}
		tos--;
		return true;
	}
	return false;
}


int main() {
	MyIntStack a(10);
	a.push(10);
	a.push(20);
	MyIntStack b = a; //복사 생성
	b.push(30);

	int n;
	a.pop(n);
	cout << "스택 a에서 팝한 값" << n << endl;
	b.pop(n);
	cout << "스택 b에서 팝한 값" << n << endl;
	b.push(40);
	b.pop(n);
	cout << "스택 b에서 팝한 값" << n << endl;
	b.pop(n);
	cout << "스택 b에서 팝한 값" << n << endl;
	b.pop(n);
	cout << "스택 b에서 팝한 값" << n << endl;

}