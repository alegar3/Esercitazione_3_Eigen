#include <iostream>
#include <Eigen>
#include <cmath>


Eigen::Vector2d Solutore_PLU(const Eigen::Matrix2d& A, const Eigen::Vector2d& b){
	
	Eigen::PartialPivLU<Eigen::Matrix2d> lu(A);	
	Eigen::Vector2d x = lu.solve(b);
	return x;
}

Eigen::Vector2d Solutore_QR(const Eigen::Matrix2d& A, const Eigen::Vector2d& b){
	
	Eigen::HouseholderQR<Eigen::Matrix2d> qr(A);	
	Eigen::Vector2d x = qr.solve(b);
	return x;
}

int main()
{
	Eigen::Matrix2d A1;
	A1 << 5.547001962252291e-01, -3.770900990025203e-02,
	8.320502943378437e-01, -9.992887623566787e-01;
	Eigen::Vector2d b1;
	b1 << -5.169911863249772e-01,
	1.672384680188350e-01;
	Eigen::Matrix2d A2;
	A2 << 5.547001962252291e-01,-5.540607316466765e-01,
	8.320502943378437e-01,-8.324762492991313e-01;
	Eigen::Vector2d b2;
	b2 << -6.394645785530173e-04, 
	4.259549612877223e-04;
	Eigen::Matrix2d A3;
	A3 << 5.547001962252291e-01, -5.547001955851905e-01, 
	8.320502943378437e-01,-8.320502947645361e-01;
	Eigen::Vector2d b3;
	b3 << -6.400391328043042e-10, 
	4.266924591433963e-10;
	
	Eigen::Vector2d x_sol_esatta;
	x_sol_esatta << -1.0e+0, 
	-1.0e+00;
	
	std::cout << "errori con LU" << std::endl;
	
	Eigen::Vector2d xLU1 = Solutore_PLU(A1,b1);
	double err_rel_LU_1 = (xLU1-x_sol_esatta).norm()/x_sol_esatta.norm();
	std::cout << err_rel_LU_1 << std::endl;
	
	Eigen::Vector2d xLU2 = Solutore_PLU(A2,b2);
	double err_rel_LU_2 = (xLU2-x_sol_esatta).norm()/x_sol_esatta.norm();
	std::cout << err_rel_LU_2 << std::endl;
	
	Eigen::Vector2d xLU3 = Solutore_PLU(A3,b3);
	double err_rel_LU_3 = (xLU3-x_sol_esatta).norm()/x_sol_esatta.norm();
	std::cout << err_rel_LU_3 << std::endl;
	
	std::cout << "errori con QR" << std::endl;
	
	Eigen::Vector2d xQR1 = Solutore_QR(A1,b1);
	double err_rel_QR_1 = (xQR1-x_sol_esatta).norm()/x_sol_esatta.norm();
	std::cout << err_rel_QR_1 << std::endl;
	
	Eigen::Vector2d xQR2 = Solutore_QR(A2,b2);
	double err_rel_QR_2 = (xQR2-x_sol_esatta).norm()/x_sol_esatta.norm();
	std::cout << err_rel_QR_2 << std::endl;
	
	Eigen::Vector2d xQR3 = Solutore_QR(A3,b3);
	double err_rel_QR_3 = (xQR3-x_sol_esatta).norm()/x_sol_esatta.norm();
	std::cout << err_rel_QR_3 << std::endl;
	
	
	
    return 0;
}
