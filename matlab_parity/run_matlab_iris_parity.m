function run_matlab_iris_parity(input_path, output_path)
% Run the authoritative deterministic multiclass algebra on a supplied split.
% Requires CVX and a licensed MOSEK installation, like the original program.

payload = load(input_path);
DATAtrain = payload.DATAtrain;
DATAtest = payload.DATAtest;
vectornu = payload.vectornu;
kernel_kind = string(payload.kernel_kind);
alpha = payload.alpha;
d = payload.polynomial_degree;
c = payload.polynomial_offset;
num_points = payload.grid_points;

dati = DATAtrain(1:end-1,:);
y_label = DATAtrain(end,:)';
m_train_tot = size(DATAtrain,2);
m_test_tot = size(DATAtest,2);
L = max(y_label);

if kernel_kind == "poly"
    K = (dati' * dati + c).^d;
elseif kernel_kind == "rbf"
    sq = sum((permute(dati,[2,3,1])-permute(dati,[3,2,1])).^2,3);
    K = exp(-sq/(2*alpha^2));
else
    error('Unknown kernel kind: %s', kernel_kind);
end

n_nu = length(vectornu);
u_by_nu = zeros(m_train_tot,L,n_nu);
gamma_by_nu = zeros(L,n_nu);
xi_by_nu = zeros(m_train_tot,L,n_nu);
b_by_nu = zeros(L,n_nu);
objective_by_nu = zeros(L,n_nu);
training_error_by_nu = zeros(L,n_nu);
raw_left_by_nu = zeros(L,n_nu);
raw_right_by_nu = zeros(L,n_nu);
selected_nu_index_1based = zeros(1,L);
u_vect = zeros(m_train_tot,L);
b_vect = zeros(1,L);
D_hat_tensore = zeros(m_train_tot,m_train_tot,L);

for l = 1:L
    y_hat = 1*(y_label==l)-1*(y_label~=l);
    D_hat = diag(y_hat);
    training_error_opt = Inf;
    for i_nu = 1:n_nu
        nu = vectornu(i_nu);
        cvx_begin quiet
            cvx_solver mosek
            cvx_precision high
            variables u_l(m_train_tot) vargamma_l xi_l(m_train_tot) s_l(m_train_tot)
            minimize sum(s_l) + nu*sum(xi_l)
            subject to
                D_hat*(K*D_hat*u_l-ones(m_train_tot,1)*vargamma_l)+xi_l >= ones(m_train_tot,1);
                xi_l >= 0;
                u_l >= -s_l;
                u_l <= s_l;
                s_l >= 0;
        cvx_end

        omega_minus_l = -min(D_hat*xi_l);
        omega_l = max(D_hat*xi_l);
        raw_left = vargamma_l+1-omega_minus_l;
        raw_right = vargamma_l-1+omega_l;
        discr_b_l = linspace(raw_left,raw_right,num_points);
        max_b = m_train_tot;
        b_opt_l = NaN;
        for j = 1:length(discr_b_l)
            count = sum(D_hat*(-K*D_hat*u_l+ones(m_train_tot,1)*discr_b_l(j))>0);
            if count < max_b
                max_b = count;
                b_opt_l = discr_b_l(j);
            end
        end
        training_error = max_b/m_train_tot;
        u_by_nu(:,l,i_nu) = u_l;
        gamma_by_nu(l,i_nu) = vargamma_l;
        xi_by_nu(:,l,i_nu) = xi_l;
        b_by_nu(l,i_nu) = b_opt_l;
        objective_by_nu(l,i_nu) = cvx_optval;
        training_error_by_nu(l,i_nu) = training_error;
        raw_left_by_nu(l,i_nu) = raw_left;
        raw_right_by_nu(l,i_nu) = raw_right;
        if training_error < training_error_opt
            training_error_opt = training_error;
            selected_nu_index_1based(l) = i_nu;
            u_vect(:,l) = u_l;
            b_vect(l) = b_opt_l;
        end
    end
    D_hat_tensore(:,:,l) = D_hat;
end

test_scores = zeros(m_test_tot,L);
prediction = zeros(m_test_tot,1);
score_tie_count = 0;
for j_test = 1:m_test_tot
    x_test = DATAtest(1:end-1,j_test);
    if kernel_kind == "poly"
        K_test = (dati'*x_test+c).^d;
    else
        K_test = exp(-sum((dati-x_test).^2,1)'/(2*alpha^2));
    end
    for l = 1:L
        test_scores(j_test,l) = K_test'*D_hat_tensore(:,:,l)*u_vect(:,l)-b_vect(l);
    end
    tied = find(test_scores(j_test,:)==max(test_scores(j_test,:)));
    score_tie_count = score_tie_count + (length(tied)>1);
    prediction(j_test) = tied(1); % source is identical when the maximum is unique
end

save(output_path,'K','u_by_nu','gamma_by_nu','xi_by_nu','b_by_nu', ...
    'objective_by_nu','training_error_by_nu','raw_left_by_nu', ...
    'raw_right_by_nu','selected_nu_index_1based','test_scores', ...
    'prediction','score_tie_count','-v7');
end
