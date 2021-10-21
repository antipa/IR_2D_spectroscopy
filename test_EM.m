h1 = figure(1)
clf

mu1 = (rand(1,2)-.5)*10;
sigma1 = rand(1,2)*5;
mu2 = (rand(1,2)-.5)*10;
sigma2 = rand(1,2)*5;


N = 500;

R1 = mvnrnd(mu1, sigma1, N);

x1 = R1(:,1);
y1 = R1(:,2);

R2 = mvnrnd(mu2, sigma2, N);

R = cat(1,R1,R2);

x2 = R2(:,1);
y2 = R2(:,2);
scatter(x2, y2);

scatter(x1, y1,'ro');
hold on
scatter(x2,y2,'b+');

mh1 = [mean(R(:,1)) + randn*std(R(:,1)), mean(R(:,2)) + randn*std(R(:,1))]
mh2 = [mean(R(:,1)) + randn*std(R(:,1)), mean(R(:,2)) + randn*std(R(:,2))]

sh1 = [var(R(:,1)), var(R(:,2))]
sh2 = [var(R(:,1)), var(R(:,2))]

M = 101;
K = 101;

xg = linspace(-15,15,M);
yg = linspace(-15,15,K);

[Xg, Yg] = meshgrid(xg,yg);

Rg = [Xg(:) Yg(:)];

g1 = reshape(mvnpdf(Rg, mh1, sh1),K,M);
g2 = reshape(mvnpdf(Rg, mh2, sh2),K,M);

surf(Xg, Yg, g1,'LineStyle','none',...
    'FaceColor',[1 .5 .5],...
    'FaceAlpha','flat',...
    'AlphaDataMapping','scaled',...
    'AlphaData',g1)

surf(Xg, Yg, g2,'LineStyle','none',...
    'FaceColor',[.5 .5 1],...
    'FaceAlpha','flat',...
    'AlphaDataMapping','scaled',...
    'AlphaData',g2)
%contour(Xg, Yg, g1,'LineColor','r')

Niter = 100
for n = 1:Niter
    set(0,'CurrentFigure',h1)
    clf
    scatter(x2, y2);

    scatter(x1, y1,'ro');
    hold on
    scatter(x2,y2,'b+');
    
    % Compute the probabilities of all points given mean and var
    prob1 = mvnpdf(R, mh1, sh1);
    prob2 = mvnpdf(R, mh2, sh2);
    
    % Create the subset of points for each of two peaks
    
    indicator = prob1>prob2;
    
    pts1 = R(indicator,:);
    pts2 = R(~indicator,:);
    
    
    


    % Compute mean and variance
    mh1(1) = mean(pts1(:,1));
    mh1(2) = mean(pts1(:,2));
    mh2(1) = mean(pts2(:,1));
    mh2(2) = mean(pts2(:,2));
    
    sh1(1) = var(pts1(:,1));
    sh1(2) = var(pts1(:,2));
    sh2(1) = var(pts2(:,1));
    sh2(2) = var(pts2(:,2));
    
    
    g1 = reshape(mvnpdf(Rg, mh1, sh1),K,M);
    g2 = reshape(mvnpdf(Rg, mh2, sh2),K,M);

%     surf(Xg, Yg, g1,'LineStyle','none',...
%         'FaceColor',[1 .5 .5],...
%         'FaceAlpha','flat',...
%         'AlphaDataMapping','scaled',...
%         'AlphaData',g1)
% 
%     surf(Xg, Yg, g2,'LineStyle','none',...
%         'FaceColor',[.5 .5 1],...
%         'FaceAlpha','flat',...
%         'AlphaDataMapping','scaled',...
%         'AlphaData',g2)
    contour(Xg,Yg,g1,'LineColor','k')
    contour(Xg,Yg,g2,'LineColor','r')
    

    drawnow
end




