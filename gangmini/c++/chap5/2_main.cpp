#include <iostream>
using namespace std;

void half(double &d) {
	d /= 2;

}

int main() {

	double n = 20;
	half(n);
	cout << n;

	return 0;
}