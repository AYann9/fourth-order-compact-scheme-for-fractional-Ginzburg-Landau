clear; clc; close all;
tic;

%% ==================== 参数设置 ====================
alpha = 1.5; beta = 1.5;          % 分数阶阶数
nu = 1; eta = 1;                   % (ν + iη)
kappa = 1; zeta = 2;               % (κ + iζ)
gamma = 3;                         % γ
theta = 0;                         % BDF2-θ (0 为标准 BDF2)

% 空间离散 (与论文一致)
h = 1/64;                          % 空间步长
x = -1 : h : 1;
y = -1 : h : 1;
N = length(x) - 1;                 % 边界点数
M = N - 1;                         % 内部网格点数

% 时间离散 (合理步长，总步数少)
T_final = 1;
dt = 1/20;                          % 时间步长
K = round(T_final / dt);           % 5 步
t = linspace(0, T_final, K+1);

% BDF2-θ 系数 (θ=0)
w0 = (3 - 2*theta)/2;   % 1.5
w1 = 2*(1 - theta);     % 2
w2 = (1 - 2*theta)/2;   % 0.5

c_complex = nu + 1i*eta;           % 线性项系数
d_complex = kappa + 1i*zeta;       % 非线性项系数

%% ==================== 辅助函数 ====================
% 分数阶中心差分系数 g_k^{(μ)}
function gk = gf(mu, k)
    gk = (-1)^k * gamma(mu+1) / ( gamma(mu/2 - k + 1) * gamma(mu/2 + k + 1) );
end

% 精确解 (与论文数值实验一致)
function u = exact8(x, y, t)
    u = (x+1).^4 .* (x-1).^4 .* (y+1).^4 .* (y-1).^4 .* exp(1i*t);
end

%% ==================== 构造离散矩阵 ====================
% 分数阶差分矩阵 Bx, By (稠密 Toeplitz)
Bx = zeros(M, M);
By = zeros(M, M);
for i = 1:M
    for j = 1:M
        k = i - j;
        Bx(i,j) = -1/h^alpha * gf(alpha, k);
        By(i,j) = -1/h^beta  * gf(beta,  k);
    end
end

% 紧致算子矩阵 Ax, Ay (稀疏三对角)
Ax = 1/12 * ( (12-alpha)*eye(M) + alpha/2 * (diag(ones(M-1,1),-1) + diag(ones(M-1,1),1)) );
Ay = 1/12 * ( (12-beta)*eye(M)  + beta/2  * (diag(ones(M-1,1),-1) + diag(ones(M-1,1),1)) );

%% ==================== 算子函数（内部网格） ====================
function L = A_op(U, Ax, Ay)
    L = Ax * U * Ay;
end

function L = Lambda_op(U, Ax, Ay, Bx, By)
    L = Ay * (Bx * U) + (Ax * U) * By;
end

%% ==================== 生成精确解矩阵 ====================
U_exact = zeros(N+1, N+1, K+1);
for k = 1:K+1
    for i = 1:N+1
        for j = 1:N+1
            U_exact(i,j,k) = exact8(x(i), y(j), t(k));
        end
    end
end

%% ==================== 数值解初始化 ====================
u_num = zeros(N+1, N+1, K+1);
u_num(:,:,1) = U_exact(:,:,1);      % 初值

%% ==================== 第一步 (k=1) 直接法 ====================
fprintf('Step 1 / %d (direct solve)\n', K);
U0_in = U_exact(2:end-1,2:end-1,1);
U1_exact_in = U_exact(2:end-1,2:end-1,2);

% 计算右端项，使精确解满足离散方程（制造解）
% 左端: A_h (u^1-u^0)/dt - c_complex Λ_h u^1 + d_complex A_h(|u^0|^2 u^0) - gamma A_h u^0
RHS_exact = (A_op(U1_exact_in, Ax, Ay) - A_op(U0_in, Ax, Ay))/dt ...
            - c_complex * Lambda_op(U1_exact_in, Ax, Ay, Bx, By) ...
            + d_complex * A_op( abs(U0_in).^2 .* U0_in, Ax, Ay ) ...
            - gamma * A_op(U0_in, Ax, Ay);

% 求解线性系统: (A_h/dt - c_complex Λ_h) u^1 = RHS_exact + A_h u^0/dt - d_complex A_h(|u^0|^2 u^0) + gamma A_h u^0
RHS1 = RHS_exact + A_op(U0_in, Ax, Ay)/dt ...
       - d_complex * A_op( abs(U0_in).^2 .* U0_in, Ax, Ay ) ...
       + gamma * A_op(U0_in, Ax, Ay);

M_mat = (1/dt) * kron(Ay, Ax) - c_complex * ( kron(Ay, Bx) + kron(By, Ax) );
u1_vec = M_mat \ reshape(RHS1, [], 1);
u_num(2:end-1, 2:end-1, 2) = reshape(u1_vec, M, M);
% 边界置零
u_num(1,:,2) = 0; u_num(end,:,2) = 0;
u_num(:,1,2) = 0; u_num(:,end,2) = 0;

%% ==================== ADI 预计算 ====================
c1 = 2*dt*(1-theta)/(3-2*theta) * c_complex;   % = 2/3 * dt * (ν+iη)
Lx = Ax - c1 * Bx;
Ly = Ay - c1 * By;
% LU 分解（复数矩阵）
[Lx_L, Lx_U] = lu(Lx);
[Ly_L, Ly_U] = lu(Ly);

%% ==================== 时间步进 (k=2 到 K) ====================
for k = 2:K
    fprintf('Step %d / %d (ADI)\n', k, K);
    u_cur  = u_num(2:end-1, 2:end-1, k);
    u_prev = u_num(2:end-1, 2:end-1, k-1);
    
    % 外推值
    u_tilde = 2*u_cur - u_prev;
    u_hat = (4*(1-theta)/(3-2*theta)) * u_cur - ((1-2*theta)/(3-2*theta)) * u_prev;
    
    % 构造右端项 (3.46)
    term1 = c_complex * theta * Lambda_op(u_cur, Ax, Ay, Bx, By);
    term2 = c_complex * (1-theta) * Lambda_op(u_hat, Ax, Ay, Bx, By);
    N1 = (1-theta) * A_op( (gamma - d_complex * abs(u_tilde).^2) .* u_tilde, Ax, Ay );
    N2 = theta * A_op( (gamma - d_complex * abs(u_cur).^2) .* u_cur, Ax, Ay );
    RHS_total = term1 + term2 + N1 + N2;
    
    % ADI 第一步: (Ax - c1 Bx) U_star = RHS_total
    U_star = zeros(M, M);
    for j = 1:M
        U_star(:,j) = Lx_U \ (Lx_L \ RHS_total(:,j));
    end
    
    % ADI 第二步: (Ay - c1 By) Psi = U_star
    Psi = zeros(M, M);
    for i = 1:M
        Psi(i,:) = (Ly_U \ (Ly_L \ U_star(i,:).')).';
    end
    
    % 更新 u^{k+1}
    u_next = (2*dt/(3-2*theta)) * Psi + (4*(1-theta)/(3-2*theta)) * u_cur - ((1-2*theta)/(3-2*theta)) * u_prev;
    
    u_num(2:end-1, 2:end-1, k+1) = u_next;
    % 边界置零
    u_num(1,:,k+1) = 0; u_num(end,:,k+1) = 0;
    u_num(:,1,k+1) = 0; u_num(:,end,k+1) = 0;
end

%% ==================== 误差计算 ====================
err = zeros(K+1, 1);
for k = 1:K+1
    diff = u_num(:,:,k) - U_exact(:,:,k);
    err(k) = h * sqrt( sum( abs(diff(:)).^2 ) );
end
max_err = max(err);
fprintf('最大 L2 误差: %e\n', max_err);

%% ==================== 绘图 ====================
figure;
semilogy(t, err, 'b-o', 'LineWidth', 1.5);
xlabel('t'); ylabel('L^2 error');
title('误差随时间演化 (BDF2-θ ADI, θ=0)');
grid on;

toc;