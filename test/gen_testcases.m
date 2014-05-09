clear all, close all                     % demoRegression and demoClassification
run('../../gpml/startup')                                  % make gpml available

mf = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [0.5; 1];
cf = {@covMaterniso, 3}; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
lf = @likGauss; sn = 0.1;                   hyp.lik = log(sn);

n = 20;
x = gpml_randn(0.3, n, 1);
K = feval(cf{:}, hyp.cov, x);
m = feval(mf{:}, hyp.mean, x);
y = chol(K)'*gpml_randn(0.15, n, 1) + m + exp(hyp.lik)*gpml_randn(0.2, n, 1);

[nlZ,dnlZ,post] = gp(hyp,[],mf,cf,lf,x,y);
save('regression', 'nlZ','dnlZ','post','x','y','mf','cf','lf','hyp')


clear all, close all

mf = @meanConst; hyp.mean = 0.25;
cf = @covSEard;  hyp.cov = log([1 0.75 0.9]);
lf = @likErf;    hyp.lik = [];

n1 = 80; n2 = 40;
S1 = eye(2); S2 = [1 0.95; 0.95 1];
m1 = [0.75; 0]; m2 = [-0.75; 0];
x1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);
x2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2);         
x = [x1 x2]';
y = [-ones(1,n1) ones(1,n2)]';

[nlZ,dnlZ,post] = gp(hyp, @infEP,mf,cf,lf,x,y);
save('classification', 'nlZ','dnlZ','post','x','y','mf','cf','lf','hyp')