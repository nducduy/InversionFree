function [w, infos] = InversionFreeNewton_LogisticRegression(problem, in_options)
% Stochastic gradient descent (SGD) algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information



    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();  

    % set local options 
    local_options = [];
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);  
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size); 

    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    % display infos
    if options.verbose > 0
        fprintf('IFN: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
    end    
    
    %Define f and beta and Q
    fIFN=100;
    betaIFN=0.75-0.5;
    betaIFN=1-0.5;
    Q=eye(d);

    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)

        for j = 1 : n
            
            % update step-size
            step = options.stepsizefun(total_iter, options);
            
            
            y       = problem.y_train(j);
            phi    = problem.x_train(:,j);
            pix=sigmoid(sum(w.*phi));
            a_n2   = max(sqrt(pix * (1 - pix)),j.^(-betaIFN));
            Phi    = a_n2 .* phi;
            fn     = Phi'* Q * Phi; 
            %theta = theta + Q * phi' .* (y - pix).*j./(j+fIFN);
            %w      = w+ Q*phi*(y-pix)*j/(j+fIFN); % for y in {0,1} 
            w      = w+ Q*sigmoid(-y*sum(w.*phi))*y*phi*j/(j+fIFN); %for y in {-1,1}
            Q      = Q - (Q * Phi * Phi' * Q)./ (1 + fn); 

            %%%%
    
           % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end  
            
            total_iter = total_iter + 1;
            
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + num_of_bachces * options.batch_size;        
        epoch = epoch + 1;

        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);        

        % display infos
        if options.verbose > 0
            fprintf('IFN: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
        end

    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epoch = %g\n', options.max_epoch);
    end
    
end