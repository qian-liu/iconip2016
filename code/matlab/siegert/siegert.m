function y=siegert(x,w,P)
% Compute the ouput of an array of Siegert Neurons.  Siegert Neurons have
% outputs that approximate the mean firing rate of leaky integrate-and-fire
% neurons with the same parameters
%
% y=siegert(x,w,P)
%
% Example Usage:
%  nInputs=5;
%  nSamples=100;
%  nUnits=3;
%  x=bsxfun(@plus,rand(nInputs,nSamples),linspace(0,500,nSamples));
%  w=.4*randn(nUnits,nInputs)+.8;
%  P=struct; P.Vth=2; P.tref=.01;
%  plot(siegert(x,w,P)'); xlabel 'input rate';ylabel 'output rate'; 
%  
%
% Input     Dim             Description
% x         [nIn,nSamples]  set of input vectors representing Poisson firing rates of inputs
% w         [nOut,nIn]      weight matrix
% P         -               structure identifying neuron parameters (see below).
%
% Ouput     Dims            Description
% y         [nOut,nSamples] set of output vectors representing Poisson output rate
%
% P is a structure array as fields of P
% Vrest         Resting Potential
% Vth           Threshold.  This can be a scalar or a vector with 1 element per unit.
% Vreset        Post-Spike Reset Potential
% taum          Membrane Time Constent
% tausyn        Synaptic response time constant.  LIF-equivalent synapic response is assumed to follow exp(x/tausyn)-exp(-x/(2*tausyn)) 
% tref          Absolute refractory Time
% normalize     Normalize output to the maximum possible firing rate (1/tref) 
% nPoints       Number of points in integral approximation
% polyapprox    Boolean indicating whether to approximate with a polynomial.  Generally it's best to leave this true.
% imagrownadult Don't waste time checking input to ensure that all inputs are positive.  The caller can do this. 
% 
% See Florian Jug's poster explaining the Siegert neuron at:
% http://www.cadmo.ethz.ch/as/people/members/fjug/personal_home/posters/2012_SiegertPoster.pdf
%
% Peter 
% oconnorp ..at.. ethz ..dot.. ch
%


%% Process Inputs

% Enable multiple input matrices (performance would be improved but code complicated by doing this later on!) 
if iscell(x)
    assert(iscell(w),'x and 2 must either both be cells or both be arrays');
    x=cell2mat(x(:));
    w=cell2mat(w(:)');
    
end

% Take in neuron arguments
assert(size(w,2)==size(x,1),'The number of columns in the weight matrix must correspond to the number of rows in the input matrix');
if ~exist('P','var'),           P=struct; end
if ~isfield(P,'Vrest'),         P.Vrest=0; end
if ~isfield(P,'Vth'),           P.Vth=1; end
if ~isfield(P,'Vreset'),        P.Vreset=0; end
if ~isfield(P,'taum'),          P.taum=0.02; end
if ~isfield(P,'tref'),          P.tref=0.002; end
if ~isfield(P,'tausyn'),        P.tausyn=0.01; end
if ~isfield(P,'imagrownadult'), P.imagrownadult=false; end
if ~isfield(P,'polyapprox'),    P.polyapprox=true; end
if ~isfield(P,'normalize'),     P.normalize=false; end

if P.polyapprox
    if isfield(P,'nPoints'),  
        assert(P.nPoints==3,'Currently the polynomial approximation can only handle nPoints=3 (Simpson''s rule).  Set nPoints to 3 or polyapprox to false.');
    else
        P.nPoints=3;
    end
else
    if ~isfield(P,'nPoints'),P.nPoints=12; end
end

k=sqrt(P.tausyn/P.taum);


assert(k<1,'taum must be greater than tausyn');

if ~P.imagrownadult
    assert(all(x(:)>=0),'x represents an input rate, and therefore cannot be negative');% People are grown-ups 
end

%% Crunch!



% Compute intermediates
Y=P.Vrest+P.taum*w*x;           % nOut x nS
G=sqrt(P.taum*w.^2*x/2);        % nOut x nS
gam=abs(zeta(.5));              % Little Gamma

% Approximate the integral
tmp=reshape(bsxfun(@plus,bsxfun(@times,meshgrid(1:P.nPoints,0:size(w,1)-1),P.Vth(:)-P.Vreset(:))/(P.nPoints-1),P.Vreset),[size(w,1) 1 P.nPoints]);
u=bsxfun(@plus, (k*gam)*G, tmp);

% u=bsxfun(@plus, (k*gam)*G, reshape(linspace(P.Vreset,P.Vth,P.nPoints),1,length(P.Vth),P.nPoints));


du=(P.Vth-P.Vreset)/(P.nPoints-1);
% uminusYoverGroot2=bsxfun(@rdivide,bsxfun(@minus,u,Y),G*sqrt(2));
% integral=du *  sum( exp(uminusYoverGroot2.^2) .* (1+erf(uminusYoverGroot2)) ,3);
YminusuoverGroot2=bsxfun(@rdivide,bsxfun(@minus,Y,u),G*sqrt(2));

% Approximate the integral.  Note: erfcx is a more numerically stable version of exp(x.^2)*erfc(x)
if P.polyapprox
    z=erfcx(YminusuoverGroot2);
    integral=bsxfun(@times,(P.Vth(:)-P.Vreset(:))/6,z(:,:,1)+4*z(:,:,2)+z(:,:,3)); % Apply Simpon's rule to approximate the integral.  This is generally pretty good as the integration curve is usually pretty close to a parabola.
else
    integral=du*sum(erfcx(YminusuoverGroot2),3); 
end

% Compute the output
y=1./(P.tref+(P.taum./G).*(sqrt(pi/2)).*integral);

if P.normalize
   y=y*P.tref; 
end


