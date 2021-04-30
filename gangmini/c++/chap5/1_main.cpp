#include <iostream>
using namespace std;

class Circle {
public: 
	int radius;
	Circle(int r=1) {
		this->radius = r;
	}
	~Circle(){}
};

void swap(Circle& a, Circle& b) {
	Circle tmp;
	tmp = a;
	a = b;
	b = tmp;
}

int main() {
	Circle c1(10);
	Circle c2(20);
	cout << c1.radius << c2.radius;
	swap(c1, c2);
	cout << c1.radius << c2.radius;
}