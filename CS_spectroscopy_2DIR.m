% I attach a sample data (?mat icon 2DIRdata_Nick.mat) and a toy code for
% the 2D IR. In the mat file, there is a 2D matrix called data_2DIR.
% It is a 2D matrix as a function of t1 and w3. The w3 (128 data point)
% axis is obtained by the spectral graph and the t1 (251 data point) is
% obtained by scanning. So we don?t need to compress w3 for now, but t1
% can be compressed first. If you run the code I attached, it will apply
% 1D fft along the t1 axis and apply the t1 to w1 conversion and plot the
% 2DIR spectrum. I used a set of data that contain many peaks, so that we
% can test the robustness of compressive sensing.
%
%
% The data_2DIR_CS is the compressed data I created, by simply randomly
% choosing 50 rows of data from the data_2DIR matrix. I also created t1_CS
% which notes the corresponding t1 delay of the chosen data in
% data_2DIR_CS.

ps_best = 0;
ps_worst = 100;
%
ps_all = []
for n = 1:1
    data=load('../data/2DIRdata_Nick.mat');
    
    %%% this part of code only deal with the FFT of the regular sampled data
    b = data.data_2DIR;
    
    
    C = ones(size(b));
    [Ny, Nx] = size(b);
    t1_CS = zeros(1,Ny);
    params.sampling = 'low_only'
    params.sampling = 'low_log'
    N_meas = 60;
    Nmax = 120;
    switch lower(params.sampling)
        case('uniform')
            rsamps = randsample(Ny, N_meas,false);
        case('dc_uniform')
            params.N_dc = 5;
            t1_CS(1:N_dc) = 1;
            N_rand = N_meas - N_dc;
            rsamps = N_dc + randsample(Ny - N_dc, N_rand,false);
        case('log')
            rsamps = unique(min(round(logspace(0,log10(Ny),N_meas)),Ny));
        case('log_rand')
            rsamps = unique(min(round(rand(1,N_meas)*3+logspace(0,log10(Ny),N_meas)),Ny));
            rsamps = cat(2,1,rsamps);
        case('subsample')
            rsamps = 1:5:Ny;
        case('low_only')
            rsamps = 1:N_meas;
        case('low_log')
            rsamps = unique(min(round(rand(1,N_meas)*5+logspace(0,log10(Nmax),N_meas)),Ny));
            rsamps = cat(2,1,rsamps);
    end
    
    t1_CS(rsamps) = 1;
    stem(t1_CS)
    axis tight
    
    %C(~ismember(data.t1,data.t1_CS),:) = 0;
    C = t1_CS'*ones(1,Nx);
    b_cs = C.*b;
    b_cs = .5*cat(1,b_cs,flipud(b_cs(2:end,:)));
    C = cat(1,C,flipud(C(2:end,:)));
    b = .5*cat(1,b,flipud(b(2:end,:)));
    
    
    spec=real(fft(b,[],1)); %FFT along t1 axis
    spec_erased = fft(b_cs,[],1);
    
    A = @(x)C.*real(ifft(x,[],1));
    A_adj = @(y)real(fft(C.*y,[],1));
    %%
    
    
    
    
    
    %calculating w1 from t1
    L=length(data.t1);
    f=1/(1e-15*2*(data.t1(2)-data.t1(1)))*linspace(-1,1,L);
    f0=1719.60;
    
    f_cm=f/2.9997e10+f0;
    
    
    
    w1=f_cm;
    %plot spectrum
    figure(1)
    clf
    subplot(2,2,1)
    h = imagesc(real(spec),'XData',data.w3,'YData',w1)
    %axis([1930 2030 1930 2030]);
    cmin = -232;
    cmax = 56;
    
    caxis([cmin cmax])
    xlabel('w3')
    ylabel('w1')
    title('Original 2D spectrum')
    
    subplot(2,2,2)
    h = imagesc(imag(spec_erased),'XData',data.w3,'YData',w1)
    %axis([1930 2030 1930 2030]);
    caxis([cmin cmax])
    xlabel('w3')
    ylabel('w1')
    title('fft of erased measurements')
    
    %% Try HQS xk = zeros(size(b));
    
    
    F = @(x)fft(x,[],1);
    Fh = @(y)ifft(y,[],1);
    params.Niter = 200;
    zkp = F(b_cs);
    
    TVpars.MAXITER = 100;
    TVpars.epsilon = 1e-6;
    loss = []
    err =[];
    resid = [];
    params.prior = 'tv';
    params.solver = 'hqs'
    params.l2_grad = false % Only used in hqs. Use l2 of gradient to smooth?
    
    
    params.tau = .0004;
    params.tau_start = .0015;
    params.dtau = (params.tau_start - params.tau)/params.Niter
    params.tau_wvlt = .002;
    params.tau_l2 = 1/sqrt(Ny);
    params.rho = .0005;
    if params.tau_start == params.tau
        tau_vec = ones(params.Niter,1)*params.tau;
    else
        tau_vec = logspace(log10(params.tau_start),...
            log10(params.tau), params.Niter);
    end
    Atb = A_adj(b_cs);
    l = zeros(size(b_cs));
    l(1,:) = 2;
    %l(1,2) = -1;
    l(2,:) = -1;
    %l(end,1) = -1;
    l(end,:) = -1;
    Lapl= abs(F(l)).^2;
    
    
    
    
    switch lower(params.prior)
        case('l1')
            grad_err_handle = @(x)linear_gradient(x,A,A_adj,b_cs);
            prox = @(x,t)soft(x,t)
            prox_handle = @(x,t)deal(soft(x,params.tau),norm(x(:),1));
            nrm = @(x)norm(x(:),1)
        case('tv')
            prox = @(x,t)TV2DFista(x, t, -3000,3000,TVpars)
            grad_err_handle = @(x)linear_gradient(x,A,A_adj,b_cs);
            nrm = @(x)TVnorm(x);
            prox_handle = @(x)deal(TV2DFista(x, params.tau, -3000,3000,TVpars), TVnorm(x))
            grad_err_handle = @(x)linear_gradient(x,A,A_adj,b_cs);
        case('l1_cplx')
            prox = @(x,t)soft_spectral(x,t);
            grad_err_handle = @(x)linear_gradient(x,A,A_adj,b_cs);
        case('l2')
            params.l2_grad = true
            grad_err_handle = @(x)deal(norm(A(x) - b_cs,'fro'),Fh((C.^2 + params.tau_l2*Lapl).*F(x)) - Atb);
            prox_handle = @(x)deal(x,0);
        case('l2_tv')
            params.l2_grad = true
            grad_err_handle = @(x)deal(norm(A(x) - b_cs,'fro'),Fh((C.^2 + params.tau_l2*Lapl).*F(x)) - Atb);
            prox = @(x,t)TV2DFista(x, t, -3000,3000,TVpars)
            prox_handle = @(x)deal(TV2DFista(x, params.tau, -3000,3000,TVpars), TVnorm(x));
        case('wvlt_tv')
            nopad = @(x)x
            wav_type = 'db9';
            grad_err_handle = @(x)linear_gradient(x,A,A_adj,b_cs);
            prox_handle = @(x)deal(.5*wavelet_detail_denoise(x,4,wav_type,[2,3],params.tau_wvlt,-500,500,nopad)+...
                .5*TV2DFista(x,params.tau,-3000,3000,TVpars),TVnorm(x));
        case('wvlt')
            nopad = @(x)x
            wav_type = 'db9';
            grad_err_handle = @(x)linear_gradient(x,A,A_adj,b_cs);
            prox_handle = @(x)deal(.5*wavelet_detail_denoise(x,4,wav_type,[3],params.tau_wvlt,-500,500,nopad),TVnorm(x));
            
    end
    
    
    
    
    
    
    
    figure(3)
    clf
    imagesc(spec)
    colorbar
    axis image
    caxis([cmin, cmax])
    
    h1 = figure(2);
    clf
    gt_spec = real(spec(1:2:end,:));
    
    
    drawnow
    options.fighandle = h1;
    options.stepsize = .95;
    options.convTol = 0;
    %options.xsize = [256,256];
    options.maxIter = params.Niter;
    options.residTol = 5e-22;
    options.momentum = 'nesterov';
    options.disp_figs = 1;
    options.disp_fig_interval = 10;   %display image this often
    options.xsize = size(b_cs);
    options.known_input = 1;
    options.print_interval = 10;
    options.xin = spec;
    ps = []
    
    switch lower(params.solver)
        case('fista')
            xinit = zkp;
            result = proxMin(grad_err_handle,prox_handle,xinit,b_cs,options);
            result_out = result(1:2:end,:);
            ps = psnr(gt_spec, result_out,max(abs(gt_spec(:))));
        case('hqs')
            
            subplot(2,2,1)
            imagesc(gt_spec(1:floor(Ny/2),:),'XData',data.w3,'YData',w1)
            caxis([cmin cmax])
            xlabel('w3')
            ylabel('w1')
            tic
            for k = 1:params.Niter
                if params.l2_grad
                    xkp = real(F((C.^2 + params.rho + params.tau_l2*Lapl).^(-1) .* Fh(F(b_cs) + params.rho*zkp)));
                else
                    xkp = real(F((C.^2 + params.rho).^(-1) .* Fh(F(b_cs) + params.rho*zkp)));
                end
                
                zkp = prox(xkp,tau_vec(k)/params.rho);
                loss = cat(1,loss,.5*norm(A(zkp) - b_cs,'fro')^2 + params.tau*nrm(zkp));
                err = cat(1,err,.5*norm(zkp - spec,'fro'));
                ps = cat(1,ps,psnr(gt_spec, zkp(1:2:end,:),max(abs(gt_spec(:)))));
                resid = cat(1,resid,norm(zkp - xkp,'fro') );
                if mod(k,options.disp_fig_interval)==0
                    subplot(2,2,2)
                    imagesc(real(zkp(1:2:Ny,:)),'XData',data.w3,'YData',w1)
                    
                    %axis([1930 2030 1930 2030]);
                    caxis([cmin cmax])
                    xlabel('w3')
                    ylabel('w1')
                    title(['iter: ', num2str(k)])
                    drawnow
                    
                    %             subplot(2,2,3)
                    %             semilogy(resid)
                    %             title('Residual norm(x-z)')
                    
                    subplot(2,2,3)
                    semilogy(loss)
                    title('loss')
                    
                    subplot(2,2,4)
                    
                    semilogy(ps)
                    title('psnr')
                end
                result_out = zkp(1:2:end,:);
            end
            toc
    end
    
    %%
    f4 = figure(4)
    clf
    
    
    %h = imagesc(spec,'XData',data.w3,'YData',w1)
    Nlevels = 30
    %gt_spec = real(spec_orig);
    
    L = linspace(min(gt_spec(:)),max(gt_spec(:)),Nlevels);
    cmin = min(L);
    cmax = max(L);
    
    no_recon_spec = F(b_cs);
    no_recon_spec = real(no_recon_spec(1:2:end,:));
    psnr_norecon = psnr(no_recon_spec,gt_spec,max(gt_spec(:)));
    subplot(2,3,1)
    contour(data.w3, w1, gt_spec,L)
    axis([1930 2030 1930 2030]);
    caxis([cmin cmax])
    xlabel('w3')
    ylabel('w1')
    title('Original 2D spectrum')
    ac = 'w'
    set(gca,'color',ac)
    set(gcf,'color','w')
    cm = 'parula'
    colormap(cm)
    
    subplot(2,3,2)
    %h = imagesc(result,'XData',data.w3,'YData',w1)
    contour(data.w3, w1, no_recon_spec,L)
    axis([1930 2030 1930 2030]);
    caxis([cmin cmax])
    set(gca,'color',ac')
    xlabel('w3')
    ylabel('w1')
    title(['No reconstruction, psnr=',num2str(psnr_norecon,'%.2f'),' dB'])
    
    subplot(2,3,3)
    %h = imagesc(result,'XData',data.w3,'YData',w1)
    contour(data.w3, w1, result_out,L)
    axis([1930 2030 1930 2030]);
    caxis([cmin cmax])
    set(gca,'color',ac')
    xlabel('w3')
    ylabel('w1')
    
    
    parfields = fieldnames(params);
    str = '';
    for n = 1:length(parfields)
        param_val = params.(parfields{n})
        if mod(n,3)==0
            esc_char = '\n';
        else
            esc_char = ', ';
        end
        if isnumeric(param_val)
            str = cat(2,str,[parfields{n},' = ',num2str(param_val,'%.1e'),esc_char]);
            
        elseif islogical(param_val)
            str = cat(2,str,[parfields{n},' = ',num2str(double(param_val),'%i'),esc_char]);
        else
            str = cat(2,str,[parfields{n},' = ',param_val,esc_char]);
        end
    end
    
    
    
    title(sprintf([str,'\n',num2str(N_meas/Ny*100,'%.1f'),...
        ' pct data, psnr = ',...
        num2str(ps(end),'%.2f'),' db']),...
        'Interpreter','none')
    
    
    colormap(cm)
    
    subplot(2,3,4)
    imagesc(real(gt_spec),'XData',data.w3,'YData',w1)
    axis([1930 2030 1930 2030]);
    caxis([cmin cmax])
    xlabel('w3')
    ylabel('w1')
    colorbar
    title('Ground truth')
    
    subplot(2,3,5)
    imagesc(no_recon_spec,'XData',data.w3,'YData',w1)
    axis([1930 2030 1930 2030]);
    caxis([cmin cmax])
    xlabel('w3')
    ylabel('w1')
    colorbar
    title('No reconstruction')
    
    subplot(2,3,6)
    imagesc(real(result_out),'XData',data.w3,'YData',w1)
    axis([1930 2030 1930 2030]);
    caxis([cmin cmax])
    xlabel('w3')
    ylabel('w1')
    colorbar
    title('Recon')
    
    if ps(end)<ps_worst
        worst_inds = rsamps;
        b_worst = b_cs;
        recon_worst = result_out;
        save('./results/worst.mat')
        export_fig(f4,'./results/worst.pdf')
        ps_worst = ps(end);
    end
    if ps(end)>ps_best
        best_inds = rsamps;
        b_best = b_cs;
        recon_best = result_out;
        save('./results/best.mat')
        export_fig(f4,'./results/best.pdf')
        ps_best = ps(end);
    end
    ps_all = cat(1,ps_all,ps(end));
end
%%
load('./results/best.mat')
f4 = figure(5)
clf


%h = imagesc(spec,'XData',data.w3,'YData',w1)
Nlevels = 30;
%gt_spec = real(spec_orig);

L = linspace(min(gt_spec(:)),max(gt_spec(:)),Nlevels);
cmin = min(L);
cmax = max(L);


no_recon_spec = recon_worst

psnr_norecon = psnr(no_recon_spec,gt_spec,max(gt_spec(:)));
subplot(2,3,1)
contour(data.w3, w1, gt_spec,L)
axis([1930 2030 1930 2030]);
caxis([cmin cmax])
xlabel('w3')
ylabel('w1')
title('Original 2D spectrum')
ac = 'w';
set(gca,'color',ac)
set(gcf,'color','w')
cm = 'parula'
colormap(cm)

subplot(2,3,2)
%h = imagesc(result,'XData',data.w3,'YData',w1)
contour(data.w3, w1, no_recon_spec,L)
axis([1930 2030 1930 2030]);
caxis([cmin cmax])
set(gca,'color',ac')
xlabel('w3')
ylabel('w1')
title(['Worst recon, psnr=',num2str(ps_worst,'%.2f'),' dB'])

subplot(2,3,3)
%h = imagesc(result,'XData',data.w3,'YData',w1)
contour(data.w3, w1, result_out,L)
axis([1930 2030 1930 2030]);
caxis([cmin cmax])
set(gca,'color',ac')
xlabel('w3')
ylabel('w1')


parfields = fieldnames(params);
str = '';
for n = 1:length(parfields)
    param_val = params.(parfields{n})
    if mod(n,3)==0
        esc_char = '\n';
    else
        esc_char = ', ';
    end
    if isnumeric(param_val)
        str = cat(2,str,[parfields{n},' = ',num2str(param_val,'%.1e'),esc_char]);

    elseif islogical(param_val)
        str = cat(2,str,[parfields{n},' = ',num2str(double(param_val),'%i'),esc_char]);
    else
        str = cat(2,str,[parfields{n},' = ',param_val,esc_char]);
    end
end



title(sprintf([str,'\n',num2str(N_meas/Ny*100,'%.1f'),...
    ' pct data, psnr = ',...
    num2str(ps(end),'%.2f'),' db']),...
    'Interpreter','none')


colormap(cm)

subplot(2,3,4)
imagesc(real(gt_spec),'XData',data.w3,'YData',w1)
axis([1930 2030 1930 2030]);
caxis([cmin cmax])
xlabel('w3')
ylabel('w1')
colorbar
title('Ground truth')

subplot(2,3,5)
imagesc(no_recon_spec,'XData',data.w3,'YData',w1)
axis([1930 2030 1930 2030]);
caxis([cmin cmax])
xlabel('w3')
ylabel('w1')
colorbar
title('No reconstruction')

subplot(2,3,6)
imagesc(real(result_out),'XData',data.w3,'YData',w1)
axis([1930 2030 1930 2030]);
caxis([cmin cmax])
xlabel('w3')
ylabel('w1')
colorbar
title('Recon')
%%

samp_worst = zeros(1,Ny);
samp_worst(worst_inds) = 1;


figure(6)
clf

subplot(2,2,2)
stem(samp_worst)
xlim([0 Nmax])
title('worst sampling')

subplot(2,2,1)
stem(t1_CS)
xlim([0 Nmax])
title('best sampling')

subplot(2,2,3)
imagesc(b_cs(1:Nmax,:))
title(['best measurement l2 norm = ',num2str(norm(b_cs,'fro'),'%.1f')])
caxis([-20 20])

subplot(2,2,4)
imagesc(b_worst(1:Nmax,:))
title(['worst measurement, l2 norm = ',num2str(norm(b_worst,'fro'),'%.1f')])
caxis([-20 20])
% legend(['worst pattern, N = ',num2str(nnz(t1_CS),'%i')],...
%     ['best pattern, N = ',num2str(nnz(t1_CS),'%i')])
hold off


