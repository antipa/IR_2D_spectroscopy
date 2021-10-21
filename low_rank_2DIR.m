clear all
data_in = load('../data/DPPC_DPPE_Antipa.mat')
[Ny,Nx,Nt] = size(data_in.sorted2D_Data(:,:,2:end));

vec = @(x)x(:);
for n = 1:Nt
    image_mat(:,n) = vec(data_in.sorted2D_Data(:,:,n));
end

[U,S,V] = svds(image_mat,Nt);

s = diag(S);
figure(2)
plot(s)
title('singular values of space-time matrix')

figure(3)

imagesc(reshape(U(:,20),[Ny,Nx]))
title('First singular vector')


%%
th = prctile(s,90);
S_th = S.*(S>th);

mat_lowrank = U*S_th*V';
for n= 1:Nt
    stack_lowrank(:,:,n) = reshape(mat_lowrank(:,n),[Ny,Nx]);
end

s_th = s.*(s>th);
h4 = figure(4)
clf
for t = 1:Nt
    set(0,'CurrentFigure',h4)


    subplot(1,2,1)
    imagesc(data_in.sorted2D_Data(:,:,t),'XData',data_in.freqAx,'YData',data_in.freqAx)
    title(sprintf('Original, T2 = %.2f',data_in.t2.listDelays(t)))
    

    subplot(1,2,2)
    imagesc(reshape(mat_lowrank(:,t),[Ny,Nx]),'XData',data_in.freqAx,'YData',data_in.freqAx);
    title(sprintf('Rank %i approx, T2 = %.2f',nnz(s_th),data_in.t2.listDelays(t)))
    drawnow
end

figure(5)
clf
subplot(1,2,1)
plot(squeeze(data_in.sorted2D_Data(134,122,:)))
xlabel('frame')
title(sprintf('Original'))

subplot(1,2,2)
plot(squeeze(stack_lowrank(134,122,:)))
xlabel('frame')
title(sprintf('Rank %i approx'))
