local opts = require 'opts'
local opt = opts.parse(arg)
local nnlib = require('cudnn')
local conv = nnlib.SpatialConvolution
local batchnorm = nnlib.SpatialBatchNormalization
local relu = nnlib.ReLU
local linear = nn.Linear
local sigmoid = nn.Sigmoid

-- Main convolutional block
local function convBlock(numIn,numOut)
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu(true))
        :add(conv(numIn,numIn/2,1,1))
        :add(batchnorm(numIn/2))
        :add(relu(true))
        :add(conv(numIn/2,numIn/2,3,3,1,1,1,1))
        :add(batchnorm(numIn/2))
        :add(relu(true))
        :add(conv(numIn/2,numOut,1,1))
end

local function transLayer(numIn,numOut,size)
    return nn.Sequential()
        :add(conv(numIn,numOut,1,1))
        :add(nn.View(opt.batchSize,-1,size*size))

end

-- Skip layer
local function attention(numIn,numOut,size)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(transLayer(numIn,numOut,size))
            :add(transLayer(numIn,numOut,size)))
        :add(nn.MM(true,false))
        :add(sigmoid(true))
end


-- Skip layer
local function skipLayer(numIn,numOut)
    if numIn == numOut then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(conv(numIn,numOut,1,1))
    end
end

-- AResidual block
function AResidual_block(numIn,numOut,size)
    return nn.Sequential()
        --:add(convBlock(numIn,numOut))
        :add(nn.ConcatTable()
            :add(transLayer(numIn,numOut,size))
            :add(attention(numIn,numOut,size)))
        :add(nn.MM(false,false))
        :add(nn.View(opt.batchSize,-1,size,size))
end

function SARM_B(numIn,numOut,size)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut))
            :add(AResidual_block(numIn,numOut,size)))
        :add(nn.CAddTable(true))
end

return SARM_B
