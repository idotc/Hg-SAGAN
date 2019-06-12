-- Pyramid Residual Module
local opts = require 'opts'
local opt = opts.parse(arg)
local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local nngraph = require 'nngraph'
local Residual = require('models.layers.Residual')
local nnlib = cudnn
paths.dofile('layers/SARM_B.lua')

local function hourglass(n, f, inp)
    -- Upper branch
    local up1 = inp
    for i = 1,opt.nResidual do up1 = Residual(f,f)(up1) end

    -- Lower branch
    local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    for i = 1,opt.nResidual do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglass(n-1,f,low1)
    else
        low2 = low1
        for i = 1,opt.nResidual do low2 = Residual(f,f)(low2) end
    end

    local low3 = low2
    for i = 1,opt.nResidual do low3 = Residual(f,f)(low3) end
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nnlib.SpatialBatchNormalization(numOut)(l))
end


local function preact(num, inp)
    return nnlib.ReLU(true)(nnlib.SpatialBatchNormalization(num)(inp))
end

function createModel(opt)

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,opt.nFeats)(r4)

    local out = {}
    local inter = r5

    for i = 1,opt.nStack do
        local hg = hourglass(4,opt.nFeats,inter)

        -- Residual layers at output resolution
        local ll = hg
        for j = 1,opt.nResidual do ll = Residual(opt.nFeats,opt.nFeats)(ll) end
        -- Linear layer to produce first set of predictions
        ll = lin(opt.nFeats,opt.nFeats,ll)
        local tmpOut
        -- Predicted heatmaps
        if i<2 then
            tmpOut = nnlib.SpatialConvolution(opt.nFeats,opt.nClasses,1,1,1,1,0,0)(ll)
        end
        if i>1 then
            tmpOut = SARM_B(opt.nFeats,opt.nClasses, 64)(ll)
            tmpOut = nnlib.SpatialConvolution(opt.nClasses,opt.nClasses,1,1,1,1,0,0)(tmpOut)
        end

        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll)
            local tmpOut_ = nnlib.SpatialConvolution(opt.nClasses,opt.nFeats,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    -- Final model
    local model = nn.gModule({inp}, out)

    return model:cuda()

end

return createModel
