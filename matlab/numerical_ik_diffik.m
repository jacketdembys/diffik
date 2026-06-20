%% numerical_ik_diffik.m
%  Numerical-IK baselines (SD / SVF / MX) for the DiffIK comparison.
%  Reads a CSV of joint vectors exported by scripts/export_for_matlab.py:
%     q1..qN   = query/GT joints  (define the target via this code's own FK)
%     qe1..qeN = example joints   = initial guess / LBE seed
%  Runs each inverse seeded from qe toward FK(q), and writes the SOLVED joints
%  back to CSV. The solved joints are then scored in Python with our FK so that
%  numerical and DiffIK use one consistent yardstick.
%
%  Reuses the validated implementations in miksolver/Numerical_Methods (no port).

clc; clear; close all;

%% --- paths (edit if your Numerical_Methods lives elsewhere) ---
NUM_METHODS = '/Users/jacketdembys/Dropbox/research/miksolver/Numerical_Methods';
addpath(genpath(NUM_METHODS));

%% --- config ---
csv_in   = 'diffik_testset.csv';
csv_out  = 'diffik_numerical_results.csv';
robot    = '7DoF-7R-Panda';          % must be known to getDH_rad
inverses = ["SD", "SVF", "MX"];
jacobian_type = 'geometric';
unit_chosen   = 1;                    % meters (Panda is all-revolute)
max_iter      = 1000;
position_error    = 0.001 * unit_chosen;   % 1 mm
orientation_error = 0.0175;                % ~1 deg
dim = 6;

%% --- load exported joints ---
Tin  = readtable(csv_in);
A    = table2array(Tin);
dof  = size(A, 2) / 2;
Q_all  = A(:, 1:dof);            % target / GT joints
Qe_all = A(:, dof+1:2*dof);      % initial guess (seed)
M = size(A, 1);
fprintf('Loaded %d samples, %d DoF, robot %s\n', M, dof, robot);

results  = [];
varNames = {};

for j = 1:length(inverses)
    inverse_chosen = inverses(j);
    fprintf('\n=== Inverse %s ===\n', inverse_chosen);
    Qsol   = zeros(M, dof);
    iters  = zeros(M, 1);
    solved = zeros(M, 1);
    times  = zeros(M, 1);

    for s = 1:M
        Q_initial = Qe_all(s, :)';
        Q_target  = Q_all(s, :)';

        % target pose via this codebase's own FK
        DH = getDH_rad(robot, Q_target, unit_chosen);
        rc = getRobotConfiguration(robot, unit_chosen, DH);
        T_final = fkine(rc, Q_target);
        D_final = getPose_rad(T_final, dim);

        % initialize at the seed
        Q_current = Q_initial;
        DH = getDH_rad(robot, Q_current, unit_chosen);
        rc = getRobotConfiguration(robot, unit_chosen, DH);
        T  = fkine(rc, Q_current);
        D_current = getPose_rad(T, dim);

        ex = D_final(1:3) - D_current(1:3);
        eo = 0.5*(cross(T(1:3,1), T_final(1:3,1)) + ...
                  cross(T(1:3,2), T_final(1:3,2)) + ...
                  cross(T(1:3,3), T_final(1:3,3)));
        e = [ex; eo];

        it = 0; final = 0; alpha = 1;
        tic;
        while final == 0
            J = getJacobianMatrix(DH, D_final, D_current, Q_current, robot, jacobian_type, unit_chosen);

            if strcmp(inverse_chosen, "SD")
                d_Q = select_dampinv2(J, 0.5, alpha * e);
            else
                inv_J = getInverseJacobian(J, inverse_chosen, e, robot);
                d_Q   = inv_J * (alpha * e);
            end

            Q_current = Q_current + d_Q;

            DH = getDH_rad(robot, Q_current, unit_chosen);
            rc = getRobotConfiguration(robot, unit_chosen, DH);
            T  = fkine(rc, Q_current);
            D_current = getPose_rad(T, dim);

            ex = D_final(1:3) - D_current(1:3);
            eo = 0.5*(cross(T(1:3,1), T_final(1:3,1)) + ...
                      cross(T(1:3,2), T_final(1:3,2)) + ...
                      cross(T(1:3,3), T_final(1:3,3)));
            e = [ex; eo];

            it = it + 1;
            reached = abs(D_final(1)-D_current(1)) < position_error && ...
                      abs(D_final(2)-D_current(2)) < position_error && ...
                      abs(D_final(3)-D_current(3)) < position_error && ...
                      abs(D_final(4)-D_current(4)) < orientation_error && ...
                      abs(D_final(5)-D_current(5)) < orientation_error && ...
                      abs(D_final(6)-D_current(6)) < orientation_error;
            if reached || it > max_iter
                final = 1;
            end
        end
        times(s)  = toc;
        Qsol(s,:) = Q_current';
        iters(s)  = it;
        solved(s) = double(it <= max_iter);

        if mod(s, 1000) == 0
            fprintf('  %s: %d/%d\n', inverse_chosen, s, M);
        end
    end

    pref = char(inverse_chosen);
    for k = 1:dof
        results = [results, Qsol(:, k)];
        varNames{end+1} = sprintf('%s_qsol%d', pref, k);
    end
    results = [results, iters, solved, times];
    varNames{end+1} = sprintf('%s_iters', pref);
    varNames{end+1} = sprintf('%s_solved', pref);
    varNames{end+1} = sprintf('%s_time', pref);
end

Tout = array2table(results, 'VariableNames', varNames);
writetable(Tout, csv_out);
fprintf('\nWrote results -> %s\n', csv_out);
