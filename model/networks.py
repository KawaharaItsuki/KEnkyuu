""""
参考:https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

"""

## インポート

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

"""
nn.Moduleを継承したクラスで、ResNetブロックを用いた生成ネットワーク

"""
class ResnetGenerator(nn.Module):
    """
    コンストラクタ
    引数:
        input_nc: 入力画像のチャンネル数
        output_nc: 出力画像のチャンネル数
        ngf: 最初の畳み込み層のフィルター数(以降の層ではこれが増加)
        norm_layer: 正規化層(デフォルト:nn.BatchNorm2d)
                            (nn.BatchNorm2d:バッチ正規化)
        use_dropout: ドロップアウト層を使用するかどうか
        n_blocks: ResNetブロックの数
        padding_type: 畳み込み層で使用するパディングの種類
                      ('reflect', 'replicate', 'zero')

    """
    def __init__(self, input_nc, output_nc, ngf = 64, norm_layer=nn.BatchNorm2d,
               use_dropout=False, n_blocks = 6, padding_type='reflect'):
        
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        """
        norm_layerがfunctools.partialで部分的に適用された関数であるかどうかチェック

        与えられたnorm_layerがバイアス項を使用するかどうかを判断する
        True:
            use_bias = True
        False:
            use_bias = False
        """
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        ## 畳み込み層、正規化層、ReLU活性化関数の適用
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        
        ## ダウンサンプリング層
        ## 畳み込み層を使用して画像の解像度を下げる
        ## 2回繰り返して画像サイズを1/4にする
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        ## ResNetブロック
        ## n_blocks個のResNetブロックを追加
        ## ResNetブロック:残差接続を持つブロック、学習を安定化させ、性能を向上させる
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        ## アップサンプリング層
        ## 逆畳み込み層を使用して画像の解像度を上げる
        ## 2回繰り返して画像サイズをもとに戻す
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        ## 最後の層
        ## Tanh活性化関数を適用して最終出力を生成
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    ## 入力を受け取り、モデルを通じて処理し、出力を生成
    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    
"""
ResNetの基本ブロックを定義している
ResNetはスキップ接続によって非常に深いネットワークを効果的に学習させることができるアーキテクチャ
ResNetの特定のブロックを定義するためのもの
"""
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    """
    dim: 畳み込み層のチャンネル数
    padding_type: パディングの種類
    norm_layer: 正規化層
    use_dropout: ドロップアウト層を使用するかどうか(boolean)
    use_bias: 畳み込み層にバイアスを使うかどうか(boolean)
    """
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    """
    畳み込みブロックを構築し、seld.conv_blockに保存
    """
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """

        ## パディングの種類に応じて適切なパディング層を追加
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        ## 畳み込み層、正規化層、ReLUを追加
        ## use_dropout=Trueのとき、ドロップアウト層を追加
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        ## 再度、パディング、畳み込み層、正規化層を追加
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        ## conv_blockをnn.Sequentailに渡し、シーケンシャルモジュールを作成
        return nn.Sequential(*conv_block)

    ## 残差接続
    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
