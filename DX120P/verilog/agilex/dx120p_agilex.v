// Copyright (c) 2024 Silicon Perception Inc (www.siliconperception.com)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


// layer    1 nstrip   14 ocmux    2 m_clk  250000000 req    37056 avail    41667 util   0.8893 nalu   1792 wmem  21233664 smem  14450688 oshape [56, 56, 256]
// layer    2 nstrip    7 ocmux    2 m_clk  250000000 req    37136 avail    41667 util   0.8913 nalu    896 wmem  10616832 smem   4014080 oshape [56, 56, 256]
// layer    3 nstrip    7 ocmux    2 m_clk  250000000 req    37136 avail    41667 util   0.8913 nalu    896 wmem  10616832 smem   4014080 oshape [56, 56, 256]
// layer    4 nstrip    7 ocmux    8 m_clk  250000000 req    74272 avail    83333 util   0.8913 nalu    224 wmem  10616832 smem   3612672 oshape [28, 28, 256]
// layer    5 nstrip    2 ocmux    2 m_clk  250000000 req    64848 avail    83333 util   0.7782 nalu    256 wmem  10616832 smem   1835008 oshape [28, 28, 256]
// layer    6 nstrip    2 ocmux    2 m_clk  250000000 req    64848 avail    83333 util   0.7782 nalu    256 wmem  10616832 smem   1835008 oshape [28, 28, 256]
// layer    7 nstrip    2 ocmux    8 m_clk  250000000 req   129696 avail   166667 util   0.7782 nalu     64 wmem  10616832 smem   1720320 oshape [14, 14, 256]
// layer    8 nstrip    1 ocmux    4 m_clk  250000000 req   129640 avail   166667 util   0.7778 nalu     64 wmem  10616832 smem    917504 oshape [14, 14, 256]
// layer    9 nstrip    1 ocmux    4 m_clk  250000000 req   129640 avail   166667 util   0.7778 nalu     64 wmem  10616832 smem    917504 oshape [14, 14, 256]
// layer   10 nstrip    1 ocmux   16 m_clk  250000000 req   259280 avail   333333 util   0.7778 nalu     16 wmem  10616832 smem    860160 oshape [7, 7, 256]
// layer   11 nstrip    1 ocmux   16 m_clk  250000000 req   185200 avail   333333 util   0.5556 nalu     16 wmem  10616832 smem    401408 oshape [5, 5, 256]
// layer   12 nstrip    1 ocmux   32 m_clk  250000000 req   222240 avail   333333 util   0.6667 nalu      8 wmem  10616832 smem    286720 oshape [3, 3, 256]
// layer   13 nstrip    1 ocmux   64 m_clk  250000000 req   148160 avail   333333 util   0.4445 nalu      4 wmem  10616832 smem    172032 oshape [1, 1, 256]
// layer   14 nstrip    1 ocmux   64 m_clk  250000000 req    17088 avail   333333 util   0.0513 nalu      8 wmem   2359296 smem    122880 oshape [1, 1, 512]

// nalu   4564
// wmem 150994944
// smem 35160064

// op    0 nstrip   14 sdepth   2016 swidth    512 wdepth   4608 wwidth   4608 bwidth   8192
// op    1 nstrip    7 sdepth    140 swidth   4096 wdepth   2304 wwidth   4608 bwidth   8192
// op    2 nstrip    7 sdepth    140 swidth   4096 wdepth   2304 wwidth   4608 bwidth   8192
// op    3 nstrip    7 sdepth    126 swidth   4096 wdepth   2304 wwidth   4608 bwidth   8192
// op    4 nstrip    2 sdepth    896 swidth   1024 wdepth   2304 wwidth   4608 bwidth   8192
// op    5 nstrip    2 sdepth    224 swidth   4096 wdepth   2304 wwidth   4608 bwidth   8192
// op    6 nstrip    2 sdepth    210 swidth   4096 wdepth   2304 wwidth   4608 bwidth   8192
// op    7 nstrip    1 sdepth    896 swidth   1024 wdepth   2304 wwidth   4608 bwidth   8192
// op    8 nstrip    1 sdepth    448 swidth   2048 wdepth   2304 wwidth   4608 bwidth   8192
// op    9 nstrip    1 sdepth    420 swidth   2048 wdepth   2304 wwidth   4608 bwidth   8192
// op   10 nstrip    1 sdepth    784 swidth    512 wdepth   2304 wwidth   4608 bwidth   8192
// op   11 nstrip    1 sdepth    560 swidth    512 wdepth   2304 wwidth   4608 bwidth   8192
// op   12 nstrip    1 sdepth    672 swidth    256 wdepth   2304 wwidth   4608 bwidth   8192
// op   13 nstrip    1 sdepth    960 swidth    128 wdepth    256 wwidth   9216 bwidth  16384

// ninst      1 sdepth    960 swidth    128
// ninst      1 sdepth    672 swidth    256
// ninst      1 sdepth    560 swidth    512
// ninst      1 sdepth    784 swidth    512
// ninst      7 sdepth    126 swidth   4096
// ninst     14 sdepth    140 swidth   4096
// ninst      2 sdepth    210 swidth   4096
// ninst      1 sdepth    420 swidth   2048
// ninst      3 sdepth    896 swidth   1024
// ninst      2 sdepth    224 swidth   4096
// ninst      1 sdepth    448 swidth   2048
// ninst     14 sdepth   2016 swidth    512

// ninst      1 wdepth    256 wwidth   9216
// ninst     12 wdepth   2304 wwidth   4608
// ninst      1 wdepth   4608 wwidth   4608

module dx120p (
    input wire clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(32):0] s_chan,
    input wire s_last,
    input wire [$clog2(112):0] s_col,
    input wire [$clog2(112):0] s_row,
    input wire [16*32-1:0] s_data,
    output wire m_valid,
    output wire [$clog2(64):0] m_chan,
    output wire m_last,
    output wire [$clog2(1):0] m_col,
    output wire [$clog2(1):0] m_row,
    output wire [8*32-1:0] m_data
);



wire op0_valid;
wire [$clog2(2):0] op0_chan;
wire op0_last;
wire [$clog2(56):0] op0_col;
wire [$clog2(56):0] op0_row;
wire [128*32-1:0] op0_data;
LAYER1 layer1 (
.reset(reset),
.s_clk(clk),
.s_valid(s_valid),
.s_chan(s_chan),
.s_last(s_last),
.s_col(s_col),
.s_row(s_row),
.s_data(s_data),
.m_clk(clk),
.m_valid(op0_valid),
.m_chan(op0_chan),
.m_last(op0_last),
.m_col(op0_col),
.m_row(op0_row),
.m_data(op0_data)
);

wire op1_valid;
wire [$clog2(2):0] op1_chan;
wire op1_last;
wire [$clog2(56):0] op1_col;
wire [$clog2(56):0] op1_row;
wire [128*32-1:0] op1_data;
LAYER2 layer2 (
.reset(reset),
.s_clk(clk),
.s_valid(op0_valid),
.s_chan(op0_chan),
.s_last(op0_last),
.s_col(op0_col),
.s_row(op0_row),
.s_data(op0_data),
.m_clk(clk),
.m_valid(op1_valid),
.m_chan(op1_chan),
.m_last(op1_last),
.m_col(op1_col),
.m_row(op1_row),
.m_data(op1_data)
);

wire op2_valid;
wire [$clog2(2):0] op2_chan;
wire op2_last;
wire [$clog2(56):0] op2_col;
wire [$clog2(56):0] op2_row;
wire [128*32-1:0] op2_data;
LAYER3 layer3 (
.reset(reset),
.s_clk(clk),
.s_valid(op1_valid),
.s_chan(op1_chan),
.s_last(op1_last),
.s_col(op1_col),
.s_row(op1_row),
.s_data(op1_data),
.m_clk(clk),
.m_valid(op2_valid),
.m_chan(op2_chan),
.m_last(op2_last),
.m_col(op2_col),
.m_row(op2_row),
.m_data(op2_data)
);

wire op3_valid;
wire [$clog2(8):0] op3_chan;
wire op3_last;
wire [$clog2(28):0] op3_col;
wire [$clog2(28):0] op3_row;
wire [32*32-1:0] op3_data;
LAYER4 layer4 (
.reset(reset),
.s_clk(clk),
.s_valid(op2_valid),
.s_chan(op2_chan),
.s_last(op2_last),
.s_col(op2_col),
.s_row(op2_row),
.s_data(op2_data),
.m_clk(clk),
.m_valid(op3_valid),
.m_chan(op3_chan),
.m_last(op3_last),
.m_col(op3_col),
.m_row(op3_row),
.m_data(op3_data)
);

wire op4_valid;
wire [$clog2(2):0] op4_chan;
wire op4_last;
wire [$clog2(28):0] op4_col;
wire [$clog2(28):0] op4_row;
wire [128*32-1:0] op4_data;
LAYER5 layer5 (
.reset(reset),
.s_clk(clk),
.s_valid(op3_valid),
.s_chan(op3_chan),
.s_last(op3_last),
.s_col(op3_col),
.s_row(op3_row),
.s_data(op3_data),
.m_clk(clk),
.m_valid(op4_valid),
.m_chan(op4_chan),
.m_last(op4_last),
.m_col(op4_col),
.m_row(op4_row),
.m_data(op4_data)
);

wire op5_valid;
wire [$clog2(2):0] op5_chan;
wire op5_last;
wire [$clog2(28):0] op5_col;
wire [$clog2(28):0] op5_row;
wire [128*32-1:0] op5_data;
LAYER6 layer6 (
.reset(reset),
.s_clk(clk),
.s_valid(op4_valid),
.s_chan(op4_chan),
.s_last(op4_last),
.s_col(op4_col),
.s_row(op4_row),
.s_data(op4_data),
.m_clk(clk),
.m_valid(op5_valid),
.m_chan(op5_chan),
.m_last(op5_last),
.m_col(op5_col),
.m_row(op5_row),
.m_data(op5_data)
);

wire op6_valid;
wire [$clog2(8):0] op6_chan;
wire op6_last;
wire [$clog2(14):0] op6_col;
wire [$clog2(14):0] op6_row;
wire [32*32-1:0] op6_data;
LAYER7 layer7 (
.reset(reset),
.s_clk(clk),
.s_valid(op5_valid),
.s_chan(op5_chan),
.s_last(op5_last),
.s_col(op5_col),
.s_row(op5_row),
.s_data(op5_data),
.m_clk(clk),
.m_valid(op6_valid),
.m_chan(op6_chan),
.m_last(op6_last),
.m_col(op6_col),
.m_row(op6_row),
.m_data(op6_data)
);

wire op7_valid;
wire [$clog2(4):0] op7_chan;
wire op7_last;
wire [$clog2(14):0] op7_col;
wire [$clog2(14):0] op7_row;
wire [64*32-1:0] op7_data;
LAYER8 layer8 (
.reset(reset),
.s_clk(clk),
.s_valid(op6_valid),
.s_chan(op6_chan),
.s_last(op6_last),
.s_col(op6_col),
.s_row(op6_row),
.s_data(op6_data),
.m_clk(clk),
.m_valid(op7_valid),
.m_chan(op7_chan),
.m_last(op7_last),
.m_col(op7_col),
.m_row(op7_row),
.m_data(op7_data)
);

wire op8_valid;
wire [$clog2(4):0] op8_chan;
wire op8_last;
wire [$clog2(14):0] op8_col;
wire [$clog2(14):0] op8_row;
wire [64*32-1:0] op8_data;
LAYER9 layer9 (
.reset(reset),
.s_clk(clk),
.s_valid(op7_valid),
.s_chan(op7_chan),
.s_last(op7_last),
.s_col(op7_col),
.s_row(op7_row),
.s_data(op7_data),
.m_clk(clk),
.m_valid(op8_valid),
.m_chan(op8_chan),
.m_last(op8_last),
.m_col(op8_col),
.m_row(op8_row),
.m_data(op8_data)
);

wire op9_valid;
wire [$clog2(16):0] op9_chan;
wire op9_last;
wire [$clog2(7):0] op9_col;
wire [$clog2(7):0] op9_row;
wire [16*32-1:0] op9_data;
LAYER10 layer10 (
.reset(reset),
.s_clk(clk),
.s_valid(op8_valid),
.s_chan(op8_chan),
.s_last(op8_last),
.s_col(op8_col),
.s_row(op8_row),
.s_data(op8_data),
.m_clk(clk),
.m_valid(op9_valid),
.m_chan(op9_chan),
.m_last(op9_last),
.m_col(op9_col),
.m_row(op9_row),
.m_data(op9_data)
);

wire op10_valid;
wire [$clog2(16):0] op10_chan;
wire op10_last;
wire [$clog2(5):0] op10_col;
wire [$clog2(5):0] op10_row;
wire [16*32-1:0] op10_data;
LAYER11 layer11 (
.reset(reset),
.s_clk(clk),
.s_valid(op9_valid),
.s_chan(op9_chan),
.s_last(op9_last),
.s_col(op9_col),
.s_row(op9_row),
.s_data(op9_data),
.m_clk(clk),
.m_valid(op10_valid),
.m_chan(op10_chan),
.m_last(op10_last),
.m_col(op10_col),
.m_row(op10_row),
.m_data(op10_data)
);

wire op11_valid;
wire [$clog2(32):0] op11_chan;
wire op11_last;
wire [$clog2(3):0] op11_col;
wire [$clog2(3):0] op11_row;
wire [8*32-1:0] op11_data;
LAYER12 layer12 (
.reset(reset),
.s_clk(clk),
.s_valid(op10_valid),
.s_chan(op10_chan),
.s_last(op10_last),
.s_col(op10_col),
.s_row(op10_row),
.s_data(op10_data),
.m_clk(clk),
.m_valid(op11_valid),
.m_chan(op11_chan),
.m_last(op11_last),
.m_col(op11_col),
.m_row(op11_row),
.m_data(op11_data)
);

wire op12_valid;
wire [$clog2(64):0] op12_chan;
wire op12_last;
wire [$clog2(1):0] op12_col;
wire [$clog2(1):0] op12_row;
wire [4*32-1:0] op12_data;
LAYER13 layer13 (
.reset(reset),
.s_clk(clk),
.s_valid(op11_valid),
.s_chan(op11_chan),
.s_last(op11_last),
.s_col(op11_col),
.s_row(op11_row),
.s_data(op11_data),
.m_clk(clk),
.m_valid(op12_valid),
.m_chan(op12_chan),
.m_last(op12_last),
.m_col(op12_col),
.m_row(op12_row),
.m_data(op12_data)
);

wire op13_valid;
wire [$clog2(64):0] op13_chan;
wire op13_last;
wire [$clog2(1):0] op13_col;
wire [$clog2(1):0] op13_row;
wire [8*32-1:0] op13_data;
LAYER14 layer14 (
.reset(reset),
.s_clk(clk),
.s_valid(op12_valid),
.s_chan(op12_chan),
.s_last(op12_last),
.s_col(op12_col),
.s_row(op12_row),
.s_data(op12_data),
.m_clk(clk),
.m_valid(op13_valid),
.m_chan(op13_chan),
.m_last(op13_last),
.m_col(op13_col),
.m_row(op13_row),
.m_data(op13_data)
);

assign m_valid=op13_valid;
assign m_chan=op13_chan;
assign m_last=op13_last;
assign m_col=op13_col;
assign m_row=op13_row;
assign m_data=op13_data;

endmodule

module LAYER1 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(32):0] s_chan,
    input wire s_last,
    input wire [$clog2(112):0] s_col,
    input wire [$clog2(112):0] s_row,
    input wire [16*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(2):0] m_chan,
    output reg m_last,
    output reg [$clog2(56):0] m_col,
    output reg [$clog2(56):0] m_row,
    output reg [128*32-1:0] m_data
);

parameter OCMUX=2,OCHAN=256,OWIDTH=56,TDMPAD=    4611;

reg [$clog2(2016):0] strip_wa [14-1:0];
reg strip_wen [14-1:0];
reg signed [$clog2(2016)+1:0] strip_ra [14-1:0];
reg signed [$clog2(4608)+1:0] weight_ra;
reg [14-1:0] strip_zpad;
reg [14-1:0] strip_zpad_q;
reg signed [$clog2(512)+1:0] ic;
reg [$clog2(512):0] ic_q,ic_qq;
reg [$clog2(2):0] ochan_sel;
reg [$clog2(14):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(32):0] s_chan_q;
reg [$clog2(112):0] s_col_q;
reg [$clog2(112):0] s_row_q;
reg [16*32-1:0] s_data_q;
reg [16*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%7)*(8+1)*32+(s_col_q-(0*8-1))*32+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*8) && (s_col_q < (0+1)*8+1-1);
    strip_wa[1] <= (s_row_q%7)*(8+1)*32+(s_col_q-(1*8-1))*32+s_chan_q;
    strip_wen[1] <= s_valid_q && (s_col_q >= 1*8-1) && (s_col_q < (1+1)*8+1-1);
    strip_wa[2] <= (s_row_q%7)*(8+1)*32+(s_col_q-(2*8-1))*32+s_chan_q;
    strip_wen[2] <= s_valid_q && (s_col_q >= 2*8-1) && (s_col_q < (2+1)*8+1-1);
    strip_wa[3] <= (s_row_q%7)*(8+1)*32+(s_col_q-(3*8-1))*32+s_chan_q;
    strip_wen[3] <= s_valid_q && (s_col_q >= 3*8-1) && (s_col_q < (3+1)*8+1-1);
    strip_wa[4] <= (s_row_q%7)*(8+1)*32+(s_col_q-(4*8-1))*32+s_chan_q;
    strip_wen[4] <= s_valid_q && (s_col_q >= 4*8-1) && (s_col_q < (4+1)*8+1-1);
    strip_wa[5] <= (s_row_q%7)*(8+1)*32+(s_col_q-(5*8-1))*32+s_chan_q;
    strip_wen[5] <= s_valid_q && (s_col_q >= 5*8-1) && (s_col_q < (5+1)*8+1-1);
    strip_wa[6] <= (s_row_q%7)*(8+1)*32+(s_col_q-(6*8-1))*32+s_chan_q;
    strip_wen[6] <= s_valid_q && (s_col_q >= 6*8-1) && (s_col_q < (6+1)*8+1-1);
    strip_wa[7] <= (s_row_q%7)*(8+1)*32+(s_col_q-(7*8-1))*32+s_chan_q;
    strip_wen[7] <= s_valid_q && (s_col_q >= 7*8-1) && (s_col_q < (7+1)*8+1-1);
    strip_wa[8] <= (s_row_q%7)*(8+1)*32+(s_col_q-(8*8-1))*32+s_chan_q;
    strip_wen[8] <= s_valid_q && (s_col_q >= 8*8-1) && (s_col_q < (8+1)*8+1-1);
    strip_wa[9] <= (s_row_q%7)*(8+1)*32+(s_col_q-(9*8-1))*32+s_chan_q;
    strip_wen[9] <= s_valid_q && (s_col_q >= 9*8-1) && (s_col_q < (9+1)*8+1-1);
    strip_wa[10] <= (s_row_q%7)*(8+1)*32+(s_col_q-(10*8-1))*32+s_chan_q;
    strip_wen[10] <= s_valid_q && (s_col_q >= 10*8-1) && (s_col_q < (10+1)*8+1-1);
    strip_wa[11] <= (s_row_q%7)*(8+1)*32+(s_col_q-(11*8-1))*32+s_chan_q;
    strip_wen[11] <= s_valid_q && (s_col_q >= 11*8-1) && (s_col_q < (11+1)*8+1-1);
    strip_wa[12] <= (s_row_q%7)*(8+1)*32+(s_col_q-(12*8-1))*32+s_chan_q;
    strip_wen[12] <= s_valid_q && (s_col_q >= 12*8-1) && (s_col_q < (12+1)*8+1-1);
    strip_wa[13] <= (s_row_q%7)*(8+1)*32+(s_col_q-(13*8-1))*32+s_chan_q;
    strip_wen[13] <= s_valid_q && (s_col_q >= 13*8-1) && (s_col_q < (13+1)*8+1-1);
end
wire [16*32-1:0] strip_rd [14-1:0];
reg [32-1:0] patch [14-1:0];
generate
for (i=0; i<14; i=i+1) begin : STRIP
STRIP #(2016,512) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(2016):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(512):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/32)*32 +:32];
    end
end
endgenerate

wire [18*256-1:0] weight_rd;
wire [32*256-1:0] bias_rd;

WEIGHT1 weight (.clk(m_clk),.ra(weight_ra[$clog2(4608):0]),.rd(weight_rd));
BIAS1 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [128-1:0];
reg [32-1:0] bias_mux [128-1:0];
generate
for (i=0; i<128; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*2+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*2+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [128*32-1:0] feat [14-1:0];
generate
for (i=0; i<14; i=i+1) begin : ALU_NS
    for (j=0; j<128; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(112):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==112-1) begin
            icount <= 'd0;
            if ((s_row_q >= 1) && ((s_row_q%2) == (3%2)))
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(3)+1:0] ky;
reg signed [$clog2(3)+1:0] kx;
reg signed [$clog2(56)+1:0] ocol;
reg signed [$clog2(56)+1:0] orow;
reg [$clog2(4611):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= ((orow==0)&&(ky<0)) || ((ocol==0)&&(kx<0));
strip_ra[0] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
strip_zpad_q[1] <= strip_zpad[1];
strip_zpad[1] <= ((orow==0)&&(ky<0));
strip_ra[1] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
strip_zpad_q[2] <= strip_zpad[2];
strip_zpad[2] <= ((orow==0)&&(ky<0));
strip_ra[2] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
strip_zpad_q[3] <= strip_zpad[3];
strip_zpad[3] <= ((orow==0)&&(ky<0));
strip_ra[3] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
strip_zpad_q[4] <= strip_zpad[4];
strip_zpad[4] <= ((orow==0)&&(ky<0));
strip_ra[4] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
strip_zpad_q[5] <= strip_zpad[5];
strip_zpad[5] <= ((orow==0)&&(ky<0));
strip_ra[5] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
strip_zpad_q[6] <= strip_zpad[6];
strip_zpad[6] <= ((orow==0)&&(ky<0));
strip_ra[6] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
strip_zpad_q[7] <= strip_zpad[7];
strip_zpad[7] <= ((orow==0)&&(ky<0));
strip_ra[7] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
strip_zpad_q[8] <= strip_zpad[8];
strip_zpad[8] <= ((orow==0)&&(ky<0));
strip_ra[8] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
strip_zpad_q[9] <= strip_zpad[9];
strip_zpad[9] <= ((orow==0)&&(ky<0));
strip_ra[9] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
strip_zpad_q[10] <= strip_zpad[10];
strip_zpad[10] <= ((orow==0)&&(ky<0));
strip_ra[10] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
strip_zpad_q[11] <= strip_zpad[11];
strip_zpad[11] <= ((orow==0)&&(ky<0));
strip_ra[11] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
strip_zpad_q[12] <= strip_zpad[12];
strip_zpad[12] <= ((orow==0)&&(ky<0));
strip_ra[12] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
strip_zpad_q[13] <= strip_zpad[13];
strip_zpad[13] <= ((orow==0)&&(ky<0));
strip_ra[13] <= ((ky+(orow*2))%7)*(8+1)*32+kx*32+ocol*2*32+(ic%32)+32;
weight_ra <= (ky+1)*3*512+(kx+1)*512+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==511) begin
                if (kx==1) begin
                    if (ky==1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd6;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd6;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd6;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd6;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<14) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*4+ocol < 56) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*4+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==2-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==2-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==4-1) begin
                        ocol <= 'd0;
                        if (orow==56-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           m_row_ack <= 1'b1;
           if (~m_row_req) begin
               m_row_ack <= 1'b0;
               m_state<= 'd0;
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS1 (
    input wire clk,
    output wire [256*32-1:0] rd
);
bias_L01 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT1 (
    input wire clk,
    input wire [$clog2(4608):0] ra,
    output wire [256*18-1:0] rd
);
weight_L01O00 u0 (
    .q(rd[0*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O01 u1 (
    .q(rd[1*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O02 u2 (
    .q(rd[2*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O03 u3 (
    .q(rd[3*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O04 u4 (
    .q(rd[4*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O05 u5 (
    .q(rd[5*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O06 u6 (
    .q(rd[6*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O07 u7 (
    .q(rd[7*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O08 u8 (
    .q(rd[8*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O09 u9 (
    .q(rd[9*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O10 u10 (
    .q(rd[10*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O11 u11 (
    .q(rd[11*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O12 u12 (
    .q(rd[12*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O13 u13 (
    .q(rd[13*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O14 u14 (
    .q(rd[14*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O15 u15 (
    .q(rd[15*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O16 u16 (
    .q(rd[16*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O17 u17 (
    .q(rd[17*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O18 u18 (
    .q(rd[18*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O19 u19 (
    .q(rd[19*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O20 u20 (
    .q(rd[20*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O21 u21 (
    .q(rd[21*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O22 u22 (
    .q(rd[22*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O23 u23 (
    .q(rd[23*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O24 u24 (
    .q(rd[24*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O25 u25 (
    .q(rd[25*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O26 u26 (
    .q(rd[26*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O27 u27 (
    .q(rd[27*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O28 u28 (
    .q(rd[28*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O29 u29 (
    .q(rd[29*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O30 u30 (
    .q(rd[30*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O31 u31 (
    .q(rd[31*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O32 u32 (
    .q(rd[32*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O33 u33 (
    .q(rd[33*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O34 u34 (
    .q(rd[34*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O35 u35 (
    .q(rd[35*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O36 u36 (
    .q(rd[36*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O37 u37 (
    .q(rd[37*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O38 u38 (
    .q(rd[38*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O39 u39 (
    .q(rd[39*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O40 u40 (
    .q(rd[40*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O41 u41 (
    .q(rd[41*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O42 u42 (
    .q(rd[42*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O43 u43 (
    .q(rd[43*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O44 u44 (
    .q(rd[44*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O45 u45 (
    .q(rd[45*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O46 u46 (
    .q(rd[46*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O47 u47 (
    .q(rd[47*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O48 u48 (
    .q(rd[48*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O49 u49 (
    .q(rd[49*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O50 u50 (
    .q(rd[50*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O51 u51 (
    .q(rd[51*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O52 u52 (
    .q(rd[52*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O53 u53 (
    .q(rd[53*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O54 u54 (
    .q(rd[54*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O55 u55 (
    .q(rd[55*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O56 u56 (
    .q(rd[56*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O57 u57 (
    .q(rd[57*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O58 u58 (
    .q(rd[58*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O59 u59 (
    .q(rd[59*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O60 u60 (
    .q(rd[60*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O61 u61 (
    .q(rd[61*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O62 u62 (
    .q(rd[62*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O63 u63 (
    .q(rd[63*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O64 u64 (
    .q(rd[64*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O65 u65 (
    .q(rd[65*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O66 u66 (
    .q(rd[66*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O67 u67 (
    .q(rd[67*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O68 u68 (
    .q(rd[68*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O69 u69 (
    .q(rd[69*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O70 u70 (
    .q(rd[70*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O71 u71 (
    .q(rd[71*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O72 u72 (
    .q(rd[72*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O73 u73 (
    .q(rd[73*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O74 u74 (
    .q(rd[74*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O75 u75 (
    .q(rd[75*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O76 u76 (
    .q(rd[76*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O77 u77 (
    .q(rd[77*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O78 u78 (
    .q(rd[78*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O79 u79 (
    .q(rd[79*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O80 u80 (
    .q(rd[80*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O81 u81 (
    .q(rd[81*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O82 u82 (
    .q(rd[82*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O83 u83 (
    .q(rd[83*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O84 u84 (
    .q(rd[84*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O85 u85 (
    .q(rd[85*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O86 u86 (
    .q(rd[86*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O87 u87 (
    .q(rd[87*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O88 u88 (
    .q(rd[88*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O89 u89 (
    .q(rd[89*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O90 u90 (
    .q(rd[90*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O91 u91 (
    .q(rd[91*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O92 u92 (
    .q(rd[92*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O93 u93 (
    .q(rd[93*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O94 u94 (
    .q(rd[94*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O95 u95 (
    .q(rd[95*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O96 u96 (
    .q(rd[96*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O97 u97 (
    .q(rd[97*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O98 u98 (
    .q(rd[98*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O99 u99 (
    .q(rd[99*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O100 u100 (
    .q(rd[100*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O101 u101 (
    .q(rd[101*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O102 u102 (
    .q(rd[102*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O103 u103 (
    .q(rd[103*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O104 u104 (
    .q(rd[104*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O105 u105 (
    .q(rd[105*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O106 u106 (
    .q(rd[106*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O107 u107 (
    .q(rd[107*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O108 u108 (
    .q(rd[108*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O109 u109 (
    .q(rd[109*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O110 u110 (
    .q(rd[110*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O111 u111 (
    .q(rd[111*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O112 u112 (
    .q(rd[112*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O113 u113 (
    .q(rd[113*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O114 u114 (
    .q(rd[114*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O115 u115 (
    .q(rd[115*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O116 u116 (
    .q(rd[116*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O117 u117 (
    .q(rd[117*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O118 u118 (
    .q(rd[118*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O119 u119 (
    .q(rd[119*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O120 u120 (
    .q(rd[120*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O121 u121 (
    .q(rd[121*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O122 u122 (
    .q(rd[122*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O123 u123 (
    .q(rd[123*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O124 u124 (
    .q(rd[124*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O125 u125 (
    .q(rd[125*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O126 u126 (
    .q(rd[126*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L01O127 u127 (
    .q(rd[127*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
endmodule

module LAYER2 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(2):0] s_chan,
    input wire s_last,
    input wire [$clog2(56):0] s_col,
    input wire [$clog2(56):0] s_row,
    input wire [128*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(2):0] m_chan,
    output reg m_last,
    output reg [$clog2(56):0] m_col,
    output reg [$clog2(56):0] m_row,
    output reg [128*32-1:0] m_data
);

parameter OCMUX=2,OCHAN=256,OWIDTH=56,TDMPAD=    4531;

reg [$clog2(140):0] strip_wa [7-1:0];
reg strip_wen [7-1:0];
reg signed [$clog2(140)+1:0] strip_ra [7-1:0];
reg signed [$clog2(2304)+1:0] weight_ra;
reg [7-1:0] strip_zpad;
reg [7-1:0] strip_zpad_q;
reg signed [$clog2(256)+1:0] ic;
reg [$clog2(256):0] ic_q,ic_qq;
reg [$clog2(2):0] ochan_sel;
reg [$clog2(7):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(2):0] s_chan_q;
reg [$clog2(56):0] s_col_q;
reg [$clog2(56):0] s_row_q;
reg [128*32-1:0] s_data_q;
reg [128*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%7)*(8+2)*2+(s_col_q-(0*8-1))*2+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*8) && (s_col_q < (0+1)*8+2-1);
    strip_wa[1] <= (s_row_q%7)*(8+2)*2+(s_col_q-(1*8-1))*2+s_chan_q;
    strip_wen[1] <= s_valid_q && (s_col_q >= 1*8-1) && (s_col_q < (1+1)*8+2-1);
    strip_wa[2] <= (s_row_q%7)*(8+2)*2+(s_col_q-(2*8-1))*2+s_chan_q;
    strip_wen[2] <= s_valid_q && (s_col_q >= 2*8-1) && (s_col_q < (2+1)*8+2-1);
    strip_wa[3] <= (s_row_q%7)*(8+2)*2+(s_col_q-(3*8-1))*2+s_chan_q;
    strip_wen[3] <= s_valid_q && (s_col_q >= 3*8-1) && (s_col_q < (3+1)*8+2-1);
    strip_wa[4] <= (s_row_q%7)*(8+2)*2+(s_col_q-(4*8-1))*2+s_chan_q;
    strip_wen[4] <= s_valid_q && (s_col_q >= 4*8-1) && (s_col_q < (4+1)*8+2-1);
    strip_wa[5] <= (s_row_q%7)*(8+2)*2+(s_col_q-(5*8-1))*2+s_chan_q;
    strip_wen[5] <= s_valid_q && (s_col_q >= 5*8-1) && (s_col_q < (5+1)*8+2-1);
    strip_wa[6] <= (s_row_q%7)*(8+2)*2+(s_col_q-(6*8-1))*2+s_chan_q;
    strip_wen[6] <= s_valid_q && (s_col_q >= 6*8-1) && (s_col_q < (6+1)*8+2-1);
end
wire [128*32-1:0] strip_rd [7-1:0];
reg [32-1:0] patch [7-1:0];
generate
for (i=0; i<7; i=i+1) begin : STRIP
STRIP #(140,4096) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(140):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(256):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/2)*32 +:32];
    end
end
endgenerate

wire [18*256-1:0] weight_rd;
wire [32*256-1:0] bias_rd;

WEIGHT2 weight (.clk(m_clk),.ra(weight_ra[$clog2(2304):0]),.rd(weight_rd));
BIAS2 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [128-1:0];
reg [32-1:0] bias_mux [128-1:0];
generate
for (i=0; i<128; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*2+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*2+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [128*32-1:0] feat [7-1:0];
generate
for (i=0; i<7; i=i+1) begin : ALU_NS
    for (j=0; j<128; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(56):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==56-1) begin
            icount <= 'd0;
            if (s_row_q >= 1)
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(3)+1:0] ky;
reg signed [$clog2(3)+1:0] kx;
reg signed [$clog2(56)+1:0] ocol;
reg signed [$clog2(56)+1:0] orow;
reg [$clog2(4531):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= ((orow==0)&&(ky<0)) || ((ocol==0)&&(kx<0)) || ((orow==55)&&(ky>0));
strip_ra[0] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
strip_zpad_q[1] <= strip_zpad[1];
strip_zpad[1] <= ((orow==0)&&(ky<0)) || ((orow==55)&&(ky>0));
strip_ra[1] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
strip_zpad_q[2] <= strip_zpad[2];
strip_zpad[2] <= ((orow==0)&&(ky<0)) || ((orow==55)&&(ky>0));
strip_ra[2] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
strip_zpad_q[3] <= strip_zpad[3];
strip_zpad[3] <= ((orow==0)&&(ky<0)) || ((orow==55)&&(ky>0));
strip_ra[3] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
strip_zpad_q[4] <= strip_zpad[4];
strip_zpad[4] <= ((orow==0)&&(ky<0)) || ((orow==55)&&(ky>0));
strip_ra[4] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
strip_zpad_q[5] <= strip_zpad[5];
strip_zpad[5] <= ((orow==0)&&(ky<0)) || ((orow==55)&&(ky>0));
strip_ra[5] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
strip_zpad_q[6] <= strip_zpad[6];
strip_zpad[6] <= ((orow==0)&&(ky<0)) || ((orow==55)&&(ky>0)) || ((ocol==7)&&(kx>0));
strip_ra[6] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
weight_ra <= (ky+1)*3*256+(kx+1)*256+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==255) begin
                if (kx==1) begin
                    if (ky==1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd6;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd6;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd6;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd6;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<7) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*8+ocol < 56) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*8+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==2-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==2-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==8-1) begin
                        ocol <= 'd0;
                        if (orow==56-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           if (orow==56-1) begin
               ocol <= 'd0;
               ochan_sel <= 'd0;
               alu_op <= 'd0;
               m_rowwait_count <= 0;
               m_state<= 'd10;
           end
           else begin
               m_row_ack <= 1'b1;
               if (~m_row_req) begin
                   m_row_ack <= 1'b0;
                   m_state<= 'd0;
               end
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS2 (
    input wire clk,
    output wire [256*32-1:0] rd
);
bias_L02 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT2 (
    input wire clk,
    input wire [$clog2(2304):0] ra,
    output wire [256*18-1:0] rd
);
weight_L02O00 u0 (
    .q(rd[0*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O01 u1 (
    .q(rd[1*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O02 u2 (
    .q(rd[2*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O03 u3 (
    .q(rd[3*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O04 u4 (
    .q(rd[4*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O05 u5 (
    .q(rd[5*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O06 u6 (
    .q(rd[6*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O07 u7 (
    .q(rd[7*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O08 u8 (
    .q(rd[8*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O09 u9 (
    .q(rd[9*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O10 u10 (
    .q(rd[10*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O11 u11 (
    .q(rd[11*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O12 u12 (
    .q(rd[12*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O13 u13 (
    .q(rd[13*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O14 u14 (
    .q(rd[14*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O15 u15 (
    .q(rd[15*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O16 u16 (
    .q(rd[16*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O17 u17 (
    .q(rd[17*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O18 u18 (
    .q(rd[18*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O19 u19 (
    .q(rd[19*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O20 u20 (
    .q(rd[20*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O21 u21 (
    .q(rd[21*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O22 u22 (
    .q(rd[22*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O23 u23 (
    .q(rd[23*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O24 u24 (
    .q(rd[24*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O25 u25 (
    .q(rd[25*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O26 u26 (
    .q(rd[26*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O27 u27 (
    .q(rd[27*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O28 u28 (
    .q(rd[28*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O29 u29 (
    .q(rd[29*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O30 u30 (
    .q(rd[30*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O31 u31 (
    .q(rd[31*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O32 u32 (
    .q(rd[32*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O33 u33 (
    .q(rd[33*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O34 u34 (
    .q(rd[34*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O35 u35 (
    .q(rd[35*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O36 u36 (
    .q(rd[36*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O37 u37 (
    .q(rd[37*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O38 u38 (
    .q(rd[38*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O39 u39 (
    .q(rd[39*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O40 u40 (
    .q(rd[40*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O41 u41 (
    .q(rd[41*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O42 u42 (
    .q(rd[42*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O43 u43 (
    .q(rd[43*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O44 u44 (
    .q(rd[44*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O45 u45 (
    .q(rd[45*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O46 u46 (
    .q(rd[46*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O47 u47 (
    .q(rd[47*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O48 u48 (
    .q(rd[48*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O49 u49 (
    .q(rd[49*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O50 u50 (
    .q(rd[50*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O51 u51 (
    .q(rd[51*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O52 u52 (
    .q(rd[52*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O53 u53 (
    .q(rd[53*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O54 u54 (
    .q(rd[54*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O55 u55 (
    .q(rd[55*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O56 u56 (
    .q(rd[56*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O57 u57 (
    .q(rd[57*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O58 u58 (
    .q(rd[58*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O59 u59 (
    .q(rd[59*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O60 u60 (
    .q(rd[60*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O61 u61 (
    .q(rd[61*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O62 u62 (
    .q(rd[62*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O63 u63 (
    .q(rd[63*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O64 u64 (
    .q(rd[64*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O65 u65 (
    .q(rd[65*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O66 u66 (
    .q(rd[66*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O67 u67 (
    .q(rd[67*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O68 u68 (
    .q(rd[68*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O69 u69 (
    .q(rd[69*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O70 u70 (
    .q(rd[70*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O71 u71 (
    .q(rd[71*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O72 u72 (
    .q(rd[72*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O73 u73 (
    .q(rd[73*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O74 u74 (
    .q(rd[74*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O75 u75 (
    .q(rd[75*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O76 u76 (
    .q(rd[76*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O77 u77 (
    .q(rd[77*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O78 u78 (
    .q(rd[78*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O79 u79 (
    .q(rd[79*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O80 u80 (
    .q(rd[80*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O81 u81 (
    .q(rd[81*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O82 u82 (
    .q(rd[82*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O83 u83 (
    .q(rd[83*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O84 u84 (
    .q(rd[84*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O85 u85 (
    .q(rd[85*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O86 u86 (
    .q(rd[86*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O87 u87 (
    .q(rd[87*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O88 u88 (
    .q(rd[88*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O89 u89 (
    .q(rd[89*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O90 u90 (
    .q(rd[90*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O91 u91 (
    .q(rd[91*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O92 u92 (
    .q(rd[92*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O93 u93 (
    .q(rd[93*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O94 u94 (
    .q(rd[94*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O95 u95 (
    .q(rd[95*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O96 u96 (
    .q(rd[96*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O97 u97 (
    .q(rd[97*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O98 u98 (
    .q(rd[98*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O99 u99 (
    .q(rd[99*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O100 u100 (
    .q(rd[100*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O101 u101 (
    .q(rd[101*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O102 u102 (
    .q(rd[102*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O103 u103 (
    .q(rd[103*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O104 u104 (
    .q(rd[104*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O105 u105 (
    .q(rd[105*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O106 u106 (
    .q(rd[106*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O107 u107 (
    .q(rd[107*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O108 u108 (
    .q(rd[108*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O109 u109 (
    .q(rd[109*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O110 u110 (
    .q(rd[110*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O111 u111 (
    .q(rd[111*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O112 u112 (
    .q(rd[112*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O113 u113 (
    .q(rd[113*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O114 u114 (
    .q(rd[114*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O115 u115 (
    .q(rd[115*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O116 u116 (
    .q(rd[116*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O117 u117 (
    .q(rd[117*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O118 u118 (
    .q(rd[118*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O119 u119 (
    .q(rd[119*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O120 u120 (
    .q(rd[120*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O121 u121 (
    .q(rd[121*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O122 u122 (
    .q(rd[122*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O123 u123 (
    .q(rd[123*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O124 u124 (
    .q(rd[124*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O125 u125 (
    .q(rd[125*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O126 u126 (
    .q(rd[126*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L02O127 u127 (
    .q(rd[127*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
endmodule

module LAYER3 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(2):0] s_chan,
    input wire s_last,
    input wire [$clog2(56):0] s_col,
    input wire [$clog2(56):0] s_row,
    input wire [128*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(2):0] m_chan,
    output reg m_last,
    output reg [$clog2(56):0] m_col,
    output reg [$clog2(56):0] m_row,
    output reg [128*32-1:0] m_data
);

parameter OCMUX=2,OCHAN=256,OWIDTH=56,TDMPAD=    4531;

reg [$clog2(140):0] strip_wa [7-1:0];
reg strip_wen [7-1:0];
reg signed [$clog2(140)+1:0] strip_ra [7-1:0];
reg signed [$clog2(2304)+1:0] weight_ra;
reg [7-1:0] strip_zpad;
reg [7-1:0] strip_zpad_q;
reg signed [$clog2(256)+1:0] ic;
reg [$clog2(256):0] ic_q,ic_qq;
reg [$clog2(2):0] ochan_sel;
reg [$clog2(7):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(2):0] s_chan_q;
reg [$clog2(56):0] s_col_q;
reg [$clog2(56):0] s_row_q;
reg [128*32-1:0] s_data_q;
reg [128*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%7)*(8+2)*2+(s_col_q-(0*8-1))*2+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*8) && (s_col_q < (0+1)*8+2-1);
    strip_wa[1] <= (s_row_q%7)*(8+2)*2+(s_col_q-(1*8-1))*2+s_chan_q;
    strip_wen[1] <= s_valid_q && (s_col_q >= 1*8-1) && (s_col_q < (1+1)*8+2-1);
    strip_wa[2] <= (s_row_q%7)*(8+2)*2+(s_col_q-(2*8-1))*2+s_chan_q;
    strip_wen[2] <= s_valid_q && (s_col_q >= 2*8-1) && (s_col_q < (2+1)*8+2-1);
    strip_wa[3] <= (s_row_q%7)*(8+2)*2+(s_col_q-(3*8-1))*2+s_chan_q;
    strip_wen[3] <= s_valid_q && (s_col_q >= 3*8-1) && (s_col_q < (3+1)*8+2-1);
    strip_wa[4] <= (s_row_q%7)*(8+2)*2+(s_col_q-(4*8-1))*2+s_chan_q;
    strip_wen[4] <= s_valid_q && (s_col_q >= 4*8-1) && (s_col_q < (4+1)*8+2-1);
    strip_wa[5] <= (s_row_q%7)*(8+2)*2+(s_col_q-(5*8-1))*2+s_chan_q;
    strip_wen[5] <= s_valid_q && (s_col_q >= 5*8-1) && (s_col_q < (5+1)*8+2-1);
    strip_wa[6] <= (s_row_q%7)*(8+2)*2+(s_col_q-(6*8-1))*2+s_chan_q;
    strip_wen[6] <= s_valid_q && (s_col_q >= 6*8-1) && (s_col_q < (6+1)*8+2-1);
end
wire [128*32-1:0] strip_rd [7-1:0];
reg [32-1:0] patch [7-1:0];
generate
for (i=0; i<7; i=i+1) begin : STRIP
STRIP #(140,4096) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(140):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(256):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/2)*32 +:32];
    end
end
endgenerate

wire [18*256-1:0] weight_rd;
wire [32*256-1:0] bias_rd;

WEIGHT3 weight (.clk(m_clk),.ra(weight_ra[$clog2(2304):0]),.rd(weight_rd));
BIAS3 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [128-1:0];
reg [32-1:0] bias_mux [128-1:0];
generate
for (i=0; i<128; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*2+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*2+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [128*32-1:0] feat [7-1:0];
generate
for (i=0; i<7; i=i+1) begin : ALU_NS
    for (j=0; j<128; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(56):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==56-1) begin
            icount <= 'd0;
            if (s_row_q >= 1)
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(3)+1:0] ky;
reg signed [$clog2(3)+1:0] kx;
reg signed [$clog2(56)+1:0] ocol;
reg signed [$clog2(56)+1:0] orow;
reg [$clog2(4531):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= ((orow==0)&&(ky<0)) || ((ocol==0)&&(kx<0)) || ((orow==55)&&(ky>0));
strip_ra[0] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
strip_zpad_q[1] <= strip_zpad[1];
strip_zpad[1] <= ((orow==0)&&(ky<0)) || ((orow==55)&&(ky>0));
strip_ra[1] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
strip_zpad_q[2] <= strip_zpad[2];
strip_zpad[2] <= ((orow==0)&&(ky<0)) || ((orow==55)&&(ky>0));
strip_ra[2] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
strip_zpad_q[3] <= strip_zpad[3];
strip_zpad[3] <= ((orow==0)&&(ky<0)) || ((orow==55)&&(ky>0));
strip_ra[3] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
strip_zpad_q[4] <= strip_zpad[4];
strip_zpad[4] <= ((orow==0)&&(ky<0)) || ((orow==55)&&(ky>0));
strip_ra[4] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
strip_zpad_q[5] <= strip_zpad[5];
strip_zpad[5] <= ((orow==0)&&(ky<0)) || ((orow==55)&&(ky>0));
strip_ra[5] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
strip_zpad_q[6] <= strip_zpad[6];
strip_zpad[6] <= ((orow==0)&&(ky<0)) || ((orow==55)&&(ky>0)) || ((ocol==7)&&(kx>0));
strip_ra[6] <= ((ky+(orow*1))%7)*(8+2)*2+kx*2+ocol*1*2+(ic%2)+2;
weight_ra <= (ky+1)*3*256+(kx+1)*256+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==255) begin
                if (kx==1) begin
                    if (ky==1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd6;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd6;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd6;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd6;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<7) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*8+ocol < 56) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*8+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==2-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==2-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==8-1) begin
                        ocol <= 'd0;
                        if (orow==56-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           if (orow==56-1) begin
               ocol <= 'd0;
               ochan_sel <= 'd0;
               alu_op <= 'd0;
               m_rowwait_count <= 0;
               m_state<= 'd10;
           end
           else begin
               m_row_ack <= 1'b1;
               if (~m_row_req) begin
                   m_row_ack <= 1'b0;
                   m_state<= 'd0;
               end
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS3 (
    input wire clk,
    output wire [256*32-1:0] rd
);
bias_L03 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT3 (
    input wire clk,
    input wire [$clog2(2304):0] ra,
    output wire [256*18-1:0] rd
);
weight_L03O00 u0 (
    .q(rd[0*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O01 u1 (
    .q(rd[1*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O02 u2 (
    .q(rd[2*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O03 u3 (
    .q(rd[3*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O04 u4 (
    .q(rd[4*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O05 u5 (
    .q(rd[5*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O06 u6 (
    .q(rd[6*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O07 u7 (
    .q(rd[7*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O08 u8 (
    .q(rd[8*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O09 u9 (
    .q(rd[9*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O10 u10 (
    .q(rd[10*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O11 u11 (
    .q(rd[11*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O12 u12 (
    .q(rd[12*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O13 u13 (
    .q(rd[13*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O14 u14 (
    .q(rd[14*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O15 u15 (
    .q(rd[15*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O16 u16 (
    .q(rd[16*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O17 u17 (
    .q(rd[17*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O18 u18 (
    .q(rd[18*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O19 u19 (
    .q(rd[19*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O20 u20 (
    .q(rd[20*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O21 u21 (
    .q(rd[21*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O22 u22 (
    .q(rd[22*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O23 u23 (
    .q(rd[23*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O24 u24 (
    .q(rd[24*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O25 u25 (
    .q(rd[25*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O26 u26 (
    .q(rd[26*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O27 u27 (
    .q(rd[27*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O28 u28 (
    .q(rd[28*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O29 u29 (
    .q(rd[29*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O30 u30 (
    .q(rd[30*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O31 u31 (
    .q(rd[31*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O32 u32 (
    .q(rd[32*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O33 u33 (
    .q(rd[33*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O34 u34 (
    .q(rd[34*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O35 u35 (
    .q(rd[35*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O36 u36 (
    .q(rd[36*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O37 u37 (
    .q(rd[37*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O38 u38 (
    .q(rd[38*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O39 u39 (
    .q(rd[39*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O40 u40 (
    .q(rd[40*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O41 u41 (
    .q(rd[41*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O42 u42 (
    .q(rd[42*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O43 u43 (
    .q(rd[43*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O44 u44 (
    .q(rd[44*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O45 u45 (
    .q(rd[45*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O46 u46 (
    .q(rd[46*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O47 u47 (
    .q(rd[47*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O48 u48 (
    .q(rd[48*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O49 u49 (
    .q(rd[49*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O50 u50 (
    .q(rd[50*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O51 u51 (
    .q(rd[51*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O52 u52 (
    .q(rd[52*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O53 u53 (
    .q(rd[53*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O54 u54 (
    .q(rd[54*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O55 u55 (
    .q(rd[55*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O56 u56 (
    .q(rd[56*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O57 u57 (
    .q(rd[57*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O58 u58 (
    .q(rd[58*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O59 u59 (
    .q(rd[59*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O60 u60 (
    .q(rd[60*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O61 u61 (
    .q(rd[61*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O62 u62 (
    .q(rd[62*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O63 u63 (
    .q(rd[63*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O64 u64 (
    .q(rd[64*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O65 u65 (
    .q(rd[65*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O66 u66 (
    .q(rd[66*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O67 u67 (
    .q(rd[67*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O68 u68 (
    .q(rd[68*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O69 u69 (
    .q(rd[69*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O70 u70 (
    .q(rd[70*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O71 u71 (
    .q(rd[71*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O72 u72 (
    .q(rd[72*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O73 u73 (
    .q(rd[73*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O74 u74 (
    .q(rd[74*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O75 u75 (
    .q(rd[75*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O76 u76 (
    .q(rd[76*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O77 u77 (
    .q(rd[77*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O78 u78 (
    .q(rd[78*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O79 u79 (
    .q(rd[79*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O80 u80 (
    .q(rd[80*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O81 u81 (
    .q(rd[81*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O82 u82 (
    .q(rd[82*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O83 u83 (
    .q(rd[83*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O84 u84 (
    .q(rd[84*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O85 u85 (
    .q(rd[85*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O86 u86 (
    .q(rd[86*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O87 u87 (
    .q(rd[87*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O88 u88 (
    .q(rd[88*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O89 u89 (
    .q(rd[89*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O90 u90 (
    .q(rd[90*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O91 u91 (
    .q(rd[91*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O92 u92 (
    .q(rd[92*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O93 u93 (
    .q(rd[93*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O94 u94 (
    .q(rd[94*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O95 u95 (
    .q(rd[95*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O96 u96 (
    .q(rd[96*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O97 u97 (
    .q(rd[97*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O98 u98 (
    .q(rd[98*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O99 u99 (
    .q(rd[99*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O100 u100 (
    .q(rd[100*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O101 u101 (
    .q(rd[101*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O102 u102 (
    .q(rd[102*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O103 u103 (
    .q(rd[103*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O104 u104 (
    .q(rd[104*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O105 u105 (
    .q(rd[105*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O106 u106 (
    .q(rd[106*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O107 u107 (
    .q(rd[107*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O108 u108 (
    .q(rd[108*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O109 u109 (
    .q(rd[109*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O110 u110 (
    .q(rd[110*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O111 u111 (
    .q(rd[111*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O112 u112 (
    .q(rd[112*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O113 u113 (
    .q(rd[113*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O114 u114 (
    .q(rd[114*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O115 u115 (
    .q(rd[115*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O116 u116 (
    .q(rd[116*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O117 u117 (
    .q(rd[117*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O118 u118 (
    .q(rd[118*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O119 u119 (
    .q(rd[119*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O120 u120 (
    .q(rd[120*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O121 u121 (
    .q(rd[121*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O122 u122 (
    .q(rd[122*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O123 u123 (
    .q(rd[123*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O124 u124 (
    .q(rd[124*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O125 u125 (
    .q(rd[125*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O126 u126 (
    .q(rd[126*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L03O127 u127 (
    .q(rd[127*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
endmodule

module LAYER4 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(2):0] s_chan,
    input wire s_last,
    input wire [$clog2(56):0] s_col,
    input wire [$clog2(56):0] s_row,
    input wire [128*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(8):0] m_chan,
    output reg m_last,
    output reg [$clog2(28):0] m_col,
    output reg [$clog2(28):0] m_row,
    output reg [32*32-1:0] m_data
);

parameter OCMUX=8,OCHAN=256,OWIDTH=28,TDMPAD=    9061;

reg [$clog2(126):0] strip_wa [7-1:0];
reg strip_wen [7-1:0];
reg signed [$clog2(126)+1:0] strip_ra [7-1:0];
reg signed [$clog2(2304)+1:0] weight_ra;
reg [7-1:0] strip_zpad;
reg [7-1:0] strip_zpad_q;
reg signed [$clog2(256)+1:0] ic;
reg [$clog2(256):0] ic_q,ic_qq;
reg [$clog2(8):0] ochan_sel;
reg [$clog2(7):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(2):0] s_chan_q;
reg [$clog2(56):0] s_col_q;
reg [$clog2(56):0] s_row_q;
reg [128*32-1:0] s_data_q;
reg [128*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%7)*(8+1)*2+(s_col_q-(0*8-1))*2+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*8) && (s_col_q < (0+1)*8+1-1);
    strip_wa[1] <= (s_row_q%7)*(8+1)*2+(s_col_q-(1*8-1))*2+s_chan_q;
    strip_wen[1] <= s_valid_q && (s_col_q >= 1*8-1) && (s_col_q < (1+1)*8+1-1);
    strip_wa[2] <= (s_row_q%7)*(8+1)*2+(s_col_q-(2*8-1))*2+s_chan_q;
    strip_wen[2] <= s_valid_q && (s_col_q >= 2*8-1) && (s_col_q < (2+1)*8+1-1);
    strip_wa[3] <= (s_row_q%7)*(8+1)*2+(s_col_q-(3*8-1))*2+s_chan_q;
    strip_wen[3] <= s_valid_q && (s_col_q >= 3*8-1) && (s_col_q < (3+1)*8+1-1);
    strip_wa[4] <= (s_row_q%7)*(8+1)*2+(s_col_q-(4*8-1))*2+s_chan_q;
    strip_wen[4] <= s_valid_q && (s_col_q >= 4*8-1) && (s_col_q < (4+1)*8+1-1);
    strip_wa[5] <= (s_row_q%7)*(8+1)*2+(s_col_q-(5*8-1))*2+s_chan_q;
    strip_wen[5] <= s_valid_q && (s_col_q >= 5*8-1) && (s_col_q < (5+1)*8+1-1);
    strip_wa[6] <= (s_row_q%7)*(8+1)*2+(s_col_q-(6*8-1))*2+s_chan_q;
    strip_wen[6] <= s_valid_q && (s_col_q >= 6*8-1) && (s_col_q < (6+1)*8+1-1);
end
wire [128*32-1:0] strip_rd [7-1:0];
reg [32-1:0] patch [7-1:0];
generate
for (i=0; i<7; i=i+1) begin : STRIP
STRIP #(126,4096) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(126):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(256):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/2)*32 +:32];
    end
end
endgenerate

wire [18*256-1:0] weight_rd;
wire [32*256-1:0] bias_rd;

WEIGHT4 weight (.clk(m_clk),.ra(weight_ra[$clog2(2304):0]),.rd(weight_rd));
BIAS4 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [32-1:0];
reg [32-1:0] bias_mux [32-1:0];
generate
for (i=0; i<32; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*8+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*8+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [32*32-1:0] feat [7-1:0];
generate
for (i=0; i<7; i=i+1) begin : ALU_NS
    for (j=0; j<32; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(56):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==56-1) begin
            icount <= 'd0;
            if ((s_row_q >= 1) && ((s_row_q%2) == (3%2)))
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(3)+1:0] ky;
reg signed [$clog2(3)+1:0] kx;
reg signed [$clog2(28)+1:0] ocol;
reg signed [$clog2(28)+1:0] orow;
reg [$clog2(9062):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= ((orow==0)&&(ky<0)) || ((ocol==0)&&(kx<0));
strip_ra[0] <= ((ky+(orow*2))%7)*(8+1)*2+kx*2+ocol*2*2+(ic%2)+2;
strip_zpad_q[1] <= strip_zpad[1];
strip_zpad[1] <= ((orow==0)&&(ky<0));
strip_ra[1] <= ((ky+(orow*2))%7)*(8+1)*2+kx*2+ocol*2*2+(ic%2)+2;
strip_zpad_q[2] <= strip_zpad[2];
strip_zpad[2] <= ((orow==0)&&(ky<0));
strip_ra[2] <= ((ky+(orow*2))%7)*(8+1)*2+kx*2+ocol*2*2+(ic%2)+2;
strip_zpad_q[3] <= strip_zpad[3];
strip_zpad[3] <= ((orow==0)&&(ky<0));
strip_ra[3] <= ((ky+(orow*2))%7)*(8+1)*2+kx*2+ocol*2*2+(ic%2)+2;
strip_zpad_q[4] <= strip_zpad[4];
strip_zpad[4] <= ((orow==0)&&(ky<0));
strip_ra[4] <= ((ky+(orow*2))%7)*(8+1)*2+kx*2+ocol*2*2+(ic%2)+2;
strip_zpad_q[5] <= strip_zpad[5];
strip_zpad[5] <= ((orow==0)&&(ky<0));
strip_ra[5] <= ((ky+(orow*2))%7)*(8+1)*2+kx*2+ocol*2*2+(ic%2)+2;
strip_zpad_q[6] <= strip_zpad[6];
strip_zpad[6] <= ((orow==0)&&(ky<0));
strip_ra[6] <= ((ky+(orow*2))%7)*(8+1)*2+kx*2+ocol*2*2+(ic%2)+2;
weight_ra <= (ky+1)*3*256+(kx+1)*256+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==255) begin
                if (kx==1) begin
                    if (ky==1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd6;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd6;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd6;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd6;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<7) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*4+ocol < 28) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*4+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==8-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==8-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==4-1) begin
                        ocol <= 'd0;
                        if (orow==28-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           m_row_ack <= 1'b1;
           if (~m_row_req) begin
               m_row_ack <= 1'b0;
               m_state<= 'd0;
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS4 (
    input wire clk,
    output wire [256*32-1:0] rd
);
bias_L04 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT4 (
    input wire clk,
    input wire [$clog2(2304):0] ra,
    output wire [256*18-1:0] rd
);
weight_L04O00 u0 (
    .q(rd[0*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O01 u1 (
    .q(rd[1*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O02 u2 (
    .q(rd[2*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O03 u3 (
    .q(rd[3*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O04 u4 (
    .q(rd[4*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O05 u5 (
    .q(rd[5*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O06 u6 (
    .q(rd[6*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O07 u7 (
    .q(rd[7*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O08 u8 (
    .q(rd[8*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O09 u9 (
    .q(rd[9*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O10 u10 (
    .q(rd[10*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O11 u11 (
    .q(rd[11*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O12 u12 (
    .q(rd[12*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O13 u13 (
    .q(rd[13*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O14 u14 (
    .q(rd[14*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O15 u15 (
    .q(rd[15*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O16 u16 (
    .q(rd[16*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O17 u17 (
    .q(rd[17*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O18 u18 (
    .q(rd[18*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O19 u19 (
    .q(rd[19*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O20 u20 (
    .q(rd[20*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O21 u21 (
    .q(rd[21*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O22 u22 (
    .q(rd[22*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O23 u23 (
    .q(rd[23*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O24 u24 (
    .q(rd[24*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O25 u25 (
    .q(rd[25*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O26 u26 (
    .q(rd[26*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O27 u27 (
    .q(rd[27*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O28 u28 (
    .q(rd[28*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O29 u29 (
    .q(rd[29*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O30 u30 (
    .q(rd[30*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L04O31 u31 (
    .q(rd[31*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
endmodule

module LAYER5 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(8):0] s_chan,
    input wire s_last,
    input wire [$clog2(28):0] s_col,
    input wire [$clog2(28):0] s_row,
    input wire [32*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(2):0] m_chan,
    output reg m_last,
    output reg [$clog2(28):0] m_col,
    output reg [$clog2(28):0] m_row,
    output reg [128*32-1:0] m_data
);

parameter OCMUX=2,OCHAN=256,OWIDTH=28,TDMPAD=   18485;

reg [$clog2(896):0] strip_wa [2-1:0];
reg strip_wen [2-1:0];
reg signed [$clog2(896)+1:0] strip_ra [2-1:0];
reg signed [$clog2(2304)+1:0] weight_ra;
reg [2-1:0] strip_zpad;
reg [2-1:0] strip_zpad_q;
reg signed [$clog2(256)+1:0] ic;
reg [$clog2(256):0] ic_q,ic_qq;
reg [$clog2(2):0] ochan_sel;
reg [$clog2(2):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(8):0] s_chan_q;
reg [$clog2(28):0] s_col_q;
reg [$clog2(28):0] s_row_q;
reg [32*32-1:0] s_data_q;
reg [32*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%7)*(14+2)*8+(s_col_q-(0*14-1))*8+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*14) && (s_col_q < (0+1)*14+2-1);
    strip_wa[1] <= (s_row_q%7)*(14+2)*8+(s_col_q-(1*14-1))*8+s_chan_q;
    strip_wen[1] <= s_valid_q && (s_col_q >= 1*14-1) && (s_col_q < (1+1)*14+2-1);
end
wire [32*32-1:0] strip_rd [2-1:0];
reg [32-1:0] patch [2-1:0];
generate
for (i=0; i<2; i=i+1) begin : STRIP
STRIP #(896,1024) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(896):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(256):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/8)*32 +:32];
    end
end
endgenerate

wire [18*256-1:0] weight_rd;
wire [32*256-1:0] bias_rd;

WEIGHT5 weight (.clk(m_clk),.ra(weight_ra[$clog2(2304):0]),.rd(weight_rd));
BIAS5 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [128-1:0];
reg [32-1:0] bias_mux [128-1:0];
generate
for (i=0; i<128; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*2+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*2+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [128*32-1:0] feat [2-1:0];
generate
for (i=0; i<2; i=i+1) begin : ALU_NS
    for (j=0; j<128; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(28):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==28-1) begin
            icount <= 'd0;
            if (s_row_q >= 1)
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(3)+1:0] ky;
reg signed [$clog2(3)+1:0] kx;
reg signed [$clog2(28)+1:0] ocol;
reg signed [$clog2(28)+1:0] orow;
reg [$clog2(18486):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= ((orow==0)&&(ky<0)) || ((ocol==0)&&(kx<0)) || ((orow==27)&&(ky>0));
strip_ra[0] <= ((ky+(orow*1))%7)*(14+2)*8+kx*8+ocol*1*8+(ic%8)+8;
strip_zpad_q[1] <= strip_zpad[1];
strip_zpad[1] <= ((orow==0)&&(ky<0)) || ((orow==27)&&(ky>0)) || ((ocol==13)&&(kx>0));
strip_ra[1] <= ((ky+(orow*1))%7)*(14+2)*8+kx*8+ocol*1*8+(ic%8)+8;
weight_ra <= (ky+1)*3*256+(kx+1)*256+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==255) begin
                if (kx==1) begin
                    if (ky==1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd6;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd6;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd6;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd6;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<2) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*14+ocol < 28) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*14+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==2-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==2-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==14-1) begin
                        ocol <= 'd0;
                        if (orow==28-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           if (orow==28-1) begin
               ocol <= 'd0;
               ochan_sel <= 'd0;
               alu_op <= 'd0;
               m_rowwait_count <= 0;
               m_state<= 'd10;
           end
           else begin
               m_row_ack <= 1'b1;
               if (~m_row_req) begin
                   m_row_ack <= 1'b0;
                   m_state<= 'd0;
               end
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS5 (
    input wire clk,
    output wire [256*32-1:0] rd
);
bias_L05 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT5 (
    input wire clk,
    input wire [$clog2(2304):0] ra,
    output wire [256*18-1:0] rd
);
weight_L05O00 u0 (
    .q(rd[0*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O01 u1 (
    .q(rd[1*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O02 u2 (
    .q(rd[2*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O03 u3 (
    .q(rd[3*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O04 u4 (
    .q(rd[4*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O05 u5 (
    .q(rd[5*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O06 u6 (
    .q(rd[6*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O07 u7 (
    .q(rd[7*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O08 u8 (
    .q(rd[8*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O09 u9 (
    .q(rd[9*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O10 u10 (
    .q(rd[10*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O11 u11 (
    .q(rd[11*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O12 u12 (
    .q(rd[12*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O13 u13 (
    .q(rd[13*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O14 u14 (
    .q(rd[14*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O15 u15 (
    .q(rd[15*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O16 u16 (
    .q(rd[16*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O17 u17 (
    .q(rd[17*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O18 u18 (
    .q(rd[18*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O19 u19 (
    .q(rd[19*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O20 u20 (
    .q(rd[20*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O21 u21 (
    .q(rd[21*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O22 u22 (
    .q(rd[22*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O23 u23 (
    .q(rd[23*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O24 u24 (
    .q(rd[24*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O25 u25 (
    .q(rd[25*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O26 u26 (
    .q(rd[26*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O27 u27 (
    .q(rd[27*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O28 u28 (
    .q(rd[28*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O29 u29 (
    .q(rd[29*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O30 u30 (
    .q(rd[30*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O31 u31 (
    .q(rd[31*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O32 u32 (
    .q(rd[32*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O33 u33 (
    .q(rd[33*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O34 u34 (
    .q(rd[34*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O35 u35 (
    .q(rd[35*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O36 u36 (
    .q(rd[36*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O37 u37 (
    .q(rd[37*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O38 u38 (
    .q(rd[38*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O39 u39 (
    .q(rd[39*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O40 u40 (
    .q(rd[40*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O41 u41 (
    .q(rd[41*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O42 u42 (
    .q(rd[42*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O43 u43 (
    .q(rd[43*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O44 u44 (
    .q(rd[44*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O45 u45 (
    .q(rd[45*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O46 u46 (
    .q(rd[46*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O47 u47 (
    .q(rd[47*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O48 u48 (
    .q(rd[48*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O49 u49 (
    .q(rd[49*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O50 u50 (
    .q(rd[50*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O51 u51 (
    .q(rd[51*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O52 u52 (
    .q(rd[52*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O53 u53 (
    .q(rd[53*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O54 u54 (
    .q(rd[54*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O55 u55 (
    .q(rd[55*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O56 u56 (
    .q(rd[56*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O57 u57 (
    .q(rd[57*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O58 u58 (
    .q(rd[58*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O59 u59 (
    .q(rd[59*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O60 u60 (
    .q(rd[60*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O61 u61 (
    .q(rd[61*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O62 u62 (
    .q(rd[62*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O63 u63 (
    .q(rd[63*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O64 u64 (
    .q(rd[64*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O65 u65 (
    .q(rd[65*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O66 u66 (
    .q(rd[66*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O67 u67 (
    .q(rd[67*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O68 u68 (
    .q(rd[68*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O69 u69 (
    .q(rd[69*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O70 u70 (
    .q(rd[70*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O71 u71 (
    .q(rd[71*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O72 u72 (
    .q(rd[72*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O73 u73 (
    .q(rd[73*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O74 u74 (
    .q(rd[74*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O75 u75 (
    .q(rd[75*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O76 u76 (
    .q(rd[76*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O77 u77 (
    .q(rd[77*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O78 u78 (
    .q(rd[78*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O79 u79 (
    .q(rd[79*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O80 u80 (
    .q(rd[80*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O81 u81 (
    .q(rd[81*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O82 u82 (
    .q(rd[82*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O83 u83 (
    .q(rd[83*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O84 u84 (
    .q(rd[84*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O85 u85 (
    .q(rd[85*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O86 u86 (
    .q(rd[86*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O87 u87 (
    .q(rd[87*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O88 u88 (
    .q(rd[88*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O89 u89 (
    .q(rd[89*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O90 u90 (
    .q(rd[90*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O91 u91 (
    .q(rd[91*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O92 u92 (
    .q(rd[92*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O93 u93 (
    .q(rd[93*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O94 u94 (
    .q(rd[94*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O95 u95 (
    .q(rd[95*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O96 u96 (
    .q(rd[96*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O97 u97 (
    .q(rd[97*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O98 u98 (
    .q(rd[98*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O99 u99 (
    .q(rd[99*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O100 u100 (
    .q(rd[100*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O101 u101 (
    .q(rd[101*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O102 u102 (
    .q(rd[102*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O103 u103 (
    .q(rd[103*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O104 u104 (
    .q(rd[104*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O105 u105 (
    .q(rd[105*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O106 u106 (
    .q(rd[106*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O107 u107 (
    .q(rd[107*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O108 u108 (
    .q(rd[108*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O109 u109 (
    .q(rd[109*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O110 u110 (
    .q(rd[110*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O111 u111 (
    .q(rd[111*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O112 u112 (
    .q(rd[112*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O113 u113 (
    .q(rd[113*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O114 u114 (
    .q(rd[114*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O115 u115 (
    .q(rd[115*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O116 u116 (
    .q(rd[116*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O117 u117 (
    .q(rd[117*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O118 u118 (
    .q(rd[118*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O119 u119 (
    .q(rd[119*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O120 u120 (
    .q(rd[120*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O121 u121 (
    .q(rd[121*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O122 u122 (
    .q(rd[122*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O123 u123 (
    .q(rd[123*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O124 u124 (
    .q(rd[124*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O125 u125 (
    .q(rd[125*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O126 u126 (
    .q(rd[126*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L05O127 u127 (
    .q(rd[127*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
endmodule

module LAYER6 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(2):0] s_chan,
    input wire s_last,
    input wire [$clog2(28):0] s_col,
    input wire [$clog2(28):0] s_row,
    input wire [128*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(2):0] m_chan,
    output reg m_last,
    output reg [$clog2(28):0] m_col,
    output reg [$clog2(28):0] m_row,
    output reg [128*32-1:0] m_data
);

parameter OCMUX=2,OCHAN=256,OWIDTH=28,TDMPAD=   18485;

reg [$clog2(224):0] strip_wa [2-1:0];
reg strip_wen [2-1:0];
reg signed [$clog2(224)+1:0] strip_ra [2-1:0];
reg signed [$clog2(2304)+1:0] weight_ra;
reg [2-1:0] strip_zpad;
reg [2-1:0] strip_zpad_q;
reg signed [$clog2(256)+1:0] ic;
reg [$clog2(256):0] ic_q,ic_qq;
reg [$clog2(2):0] ochan_sel;
reg [$clog2(2):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(2):0] s_chan_q;
reg [$clog2(28):0] s_col_q;
reg [$clog2(28):0] s_row_q;
reg [128*32-1:0] s_data_q;
reg [128*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%7)*(14+2)*2+(s_col_q-(0*14-1))*2+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*14) && (s_col_q < (0+1)*14+2-1);
    strip_wa[1] <= (s_row_q%7)*(14+2)*2+(s_col_q-(1*14-1))*2+s_chan_q;
    strip_wen[1] <= s_valid_q && (s_col_q >= 1*14-1) && (s_col_q < (1+1)*14+2-1);
end
wire [128*32-1:0] strip_rd [2-1:0];
reg [32-1:0] patch [2-1:0];
generate
for (i=0; i<2; i=i+1) begin : STRIP
STRIP #(224,4096) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(224):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(256):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/2)*32 +:32];
    end
end
endgenerate

wire [18*256-1:0] weight_rd;
wire [32*256-1:0] bias_rd;

WEIGHT6 weight (.clk(m_clk),.ra(weight_ra[$clog2(2304):0]),.rd(weight_rd));
BIAS6 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [128-1:0];
reg [32-1:0] bias_mux [128-1:0];
generate
for (i=0; i<128; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*2+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*2+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [128*32-1:0] feat [2-1:0];
generate
for (i=0; i<2; i=i+1) begin : ALU_NS
    for (j=0; j<128; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(28):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==28-1) begin
            icount <= 'd0;
            if (s_row_q >= 1)
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(3)+1:0] ky;
reg signed [$clog2(3)+1:0] kx;
reg signed [$clog2(28)+1:0] ocol;
reg signed [$clog2(28)+1:0] orow;
reg [$clog2(18486):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= ((orow==0)&&(ky<0)) || ((ocol==0)&&(kx<0)) || ((orow==27)&&(ky>0));
strip_ra[0] <= ((ky+(orow*1))%7)*(14+2)*2+kx*2+ocol*1*2+(ic%2)+2;
strip_zpad_q[1] <= strip_zpad[1];
strip_zpad[1] <= ((orow==0)&&(ky<0)) || ((orow==27)&&(ky>0)) || ((ocol==13)&&(kx>0));
strip_ra[1] <= ((ky+(orow*1))%7)*(14+2)*2+kx*2+ocol*1*2+(ic%2)+2;
weight_ra <= (ky+1)*3*256+(kx+1)*256+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==255) begin
                if (kx==1) begin
                    if (ky==1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd6;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd6;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd6;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd6;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<2) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*14+ocol < 28) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*14+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==2-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==2-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==14-1) begin
                        ocol <= 'd0;
                        if (orow==28-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           if (orow==28-1) begin
               ocol <= 'd0;
               ochan_sel <= 'd0;
               alu_op <= 'd0;
               m_rowwait_count <= 0;
               m_state<= 'd10;
           end
           else begin
               m_row_ack <= 1'b1;
               if (~m_row_req) begin
                   m_row_ack <= 1'b0;
                   m_state<= 'd0;
               end
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS6 (
    input wire clk,
    output wire [256*32-1:0] rd
);
bias_L06 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT6 (
    input wire clk,
    input wire [$clog2(2304):0] ra,
    output wire [256*18-1:0] rd
);
weight_L06O00 u0 (
    .q(rd[0*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O01 u1 (
    .q(rd[1*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O02 u2 (
    .q(rd[2*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O03 u3 (
    .q(rd[3*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O04 u4 (
    .q(rd[4*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O05 u5 (
    .q(rd[5*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O06 u6 (
    .q(rd[6*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O07 u7 (
    .q(rd[7*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O08 u8 (
    .q(rd[8*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O09 u9 (
    .q(rd[9*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O10 u10 (
    .q(rd[10*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O11 u11 (
    .q(rd[11*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O12 u12 (
    .q(rd[12*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O13 u13 (
    .q(rd[13*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O14 u14 (
    .q(rd[14*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O15 u15 (
    .q(rd[15*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O16 u16 (
    .q(rd[16*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O17 u17 (
    .q(rd[17*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O18 u18 (
    .q(rd[18*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O19 u19 (
    .q(rd[19*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O20 u20 (
    .q(rd[20*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O21 u21 (
    .q(rd[21*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O22 u22 (
    .q(rd[22*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O23 u23 (
    .q(rd[23*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O24 u24 (
    .q(rd[24*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O25 u25 (
    .q(rd[25*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O26 u26 (
    .q(rd[26*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O27 u27 (
    .q(rd[27*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O28 u28 (
    .q(rd[28*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O29 u29 (
    .q(rd[29*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O30 u30 (
    .q(rd[30*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O31 u31 (
    .q(rd[31*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O32 u32 (
    .q(rd[32*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O33 u33 (
    .q(rd[33*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O34 u34 (
    .q(rd[34*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O35 u35 (
    .q(rd[35*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O36 u36 (
    .q(rd[36*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O37 u37 (
    .q(rd[37*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O38 u38 (
    .q(rd[38*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O39 u39 (
    .q(rd[39*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O40 u40 (
    .q(rd[40*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O41 u41 (
    .q(rd[41*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O42 u42 (
    .q(rd[42*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O43 u43 (
    .q(rd[43*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O44 u44 (
    .q(rd[44*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O45 u45 (
    .q(rd[45*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O46 u46 (
    .q(rd[46*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O47 u47 (
    .q(rd[47*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O48 u48 (
    .q(rd[48*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O49 u49 (
    .q(rd[49*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O50 u50 (
    .q(rd[50*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O51 u51 (
    .q(rd[51*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O52 u52 (
    .q(rd[52*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O53 u53 (
    .q(rd[53*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O54 u54 (
    .q(rd[54*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O55 u55 (
    .q(rd[55*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O56 u56 (
    .q(rd[56*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O57 u57 (
    .q(rd[57*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O58 u58 (
    .q(rd[58*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O59 u59 (
    .q(rd[59*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O60 u60 (
    .q(rd[60*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O61 u61 (
    .q(rd[61*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O62 u62 (
    .q(rd[62*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O63 u63 (
    .q(rd[63*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O64 u64 (
    .q(rd[64*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O65 u65 (
    .q(rd[65*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O66 u66 (
    .q(rd[66*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O67 u67 (
    .q(rd[67*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O68 u68 (
    .q(rd[68*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O69 u69 (
    .q(rd[69*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O70 u70 (
    .q(rd[70*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O71 u71 (
    .q(rd[71*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O72 u72 (
    .q(rd[72*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O73 u73 (
    .q(rd[73*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O74 u74 (
    .q(rd[74*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O75 u75 (
    .q(rd[75*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O76 u76 (
    .q(rd[76*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O77 u77 (
    .q(rd[77*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O78 u78 (
    .q(rd[78*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O79 u79 (
    .q(rd[79*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O80 u80 (
    .q(rd[80*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O81 u81 (
    .q(rd[81*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O82 u82 (
    .q(rd[82*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O83 u83 (
    .q(rd[83*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O84 u84 (
    .q(rd[84*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O85 u85 (
    .q(rd[85*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O86 u86 (
    .q(rd[86*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O87 u87 (
    .q(rd[87*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O88 u88 (
    .q(rd[88*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O89 u89 (
    .q(rd[89*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O90 u90 (
    .q(rd[90*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O91 u91 (
    .q(rd[91*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O92 u92 (
    .q(rd[92*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O93 u93 (
    .q(rd[93*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O94 u94 (
    .q(rd[94*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O95 u95 (
    .q(rd[95*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O96 u96 (
    .q(rd[96*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O97 u97 (
    .q(rd[97*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O98 u98 (
    .q(rd[98*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O99 u99 (
    .q(rd[99*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O100 u100 (
    .q(rd[100*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O101 u101 (
    .q(rd[101*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O102 u102 (
    .q(rd[102*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O103 u103 (
    .q(rd[103*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O104 u104 (
    .q(rd[104*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O105 u105 (
    .q(rd[105*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O106 u106 (
    .q(rd[106*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O107 u107 (
    .q(rd[107*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O108 u108 (
    .q(rd[108*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O109 u109 (
    .q(rd[109*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O110 u110 (
    .q(rd[110*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O111 u111 (
    .q(rd[111*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O112 u112 (
    .q(rd[112*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O113 u113 (
    .q(rd[113*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O114 u114 (
    .q(rd[114*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O115 u115 (
    .q(rd[115*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O116 u116 (
    .q(rd[116*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O117 u117 (
    .q(rd[117*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O118 u118 (
    .q(rd[118*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O119 u119 (
    .q(rd[119*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O120 u120 (
    .q(rd[120*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O121 u121 (
    .q(rd[121*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O122 u122 (
    .q(rd[122*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O123 u123 (
    .q(rd[123*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O124 u124 (
    .q(rd[124*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O125 u125 (
    .q(rd[125*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O126 u126 (
    .q(rd[126*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
weight_L06O127 u127 (
    .q(rd[127*18*2 +:18*2]),
    .address(ra),
    .clock(clk)
);
endmodule

module LAYER7 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(2):0] s_chan,
    input wire s_last,
    input wire [$clog2(28):0] s_col,
    input wire [$clog2(28):0] s_row,
    input wire [128*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(8):0] m_chan,
    output reg m_last,
    output reg [$clog2(14):0] m_col,
    output reg [$clog2(14):0] m_row,
    output reg [32*32-1:0] m_data
);

parameter OCMUX=8,OCHAN=256,OWIDTH=14,TDMPAD=   36971;

reg [$clog2(210):0] strip_wa [2-1:0];
reg strip_wen [2-1:0];
reg signed [$clog2(210)+1:0] strip_ra [2-1:0];
reg signed [$clog2(2304)+1:0] weight_ra;
reg [2-1:0] strip_zpad;
reg [2-1:0] strip_zpad_q;
reg signed [$clog2(256)+1:0] ic;
reg [$clog2(256):0] ic_q,ic_qq;
reg [$clog2(8):0] ochan_sel;
reg [$clog2(2):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(2):0] s_chan_q;
reg [$clog2(28):0] s_col_q;
reg [$clog2(28):0] s_row_q;
reg [128*32-1:0] s_data_q;
reg [128*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%7)*(14+1)*2+(s_col_q-(0*14-1))*2+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*14) && (s_col_q < (0+1)*14+1-1);
    strip_wa[1] <= (s_row_q%7)*(14+1)*2+(s_col_q-(1*14-1))*2+s_chan_q;
    strip_wen[1] <= s_valid_q && (s_col_q >= 1*14-1) && (s_col_q < (1+1)*14+1-1);
end
wire [128*32-1:0] strip_rd [2-1:0];
reg [32-1:0] patch [2-1:0];
generate
for (i=0; i<2; i=i+1) begin : STRIP
STRIP #(210,4096) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(210):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(256):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/2)*32 +:32];
    end
end
endgenerate

wire [18*256-1:0] weight_rd;
wire [32*256-1:0] bias_rd;

WEIGHT7 weight (.clk(m_clk),.ra(weight_ra[$clog2(2304):0]),.rd(weight_rd));
BIAS7 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [32-1:0];
reg [32-1:0] bias_mux [32-1:0];
generate
for (i=0; i<32; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*8+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*8+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [32*32-1:0] feat [2-1:0];
generate
for (i=0; i<2; i=i+1) begin : ALU_NS
    for (j=0; j<32; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(28):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==28-1) begin
            icount <= 'd0;
            if ((s_row_q >= 1) && ((s_row_q%2) == (3%2)))
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(3)+1:0] ky;
reg signed [$clog2(3)+1:0] kx;
reg signed [$clog2(14)+1:0] ocol;
reg signed [$clog2(14)+1:0] orow;
reg [$clog2(36971):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= ((orow==0)&&(ky<0)) || ((ocol==0)&&(kx<0));
strip_ra[0] <= ((ky+(orow*2))%7)*(14+1)*2+kx*2+ocol*2*2+(ic%2)+2;
strip_zpad_q[1] <= strip_zpad[1];
strip_zpad[1] <= ((orow==0)&&(ky<0));
strip_ra[1] <= ((ky+(orow*2))%7)*(14+1)*2+kx*2+ocol*2*2+(ic%2)+2;
weight_ra <= (ky+1)*3*256+(kx+1)*256+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==255) begin
                if (kx==1) begin
                    if (ky==1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd6;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd6;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd6;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd6;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<2) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*7+ocol < 14) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*7+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==8-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==8-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==7-1) begin
                        ocol <= 'd0;
                        if (orow==14-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           m_row_ack <= 1'b1;
           if (~m_row_req) begin
               m_row_ack <= 1'b0;
               m_state<= 'd0;
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS7 (
    input wire clk,
    output wire [256*32-1:0] rd
);
bias_L07 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT7 (
    input wire clk,
    input wire [$clog2(2304):0] ra,
    output wire [256*18-1:0] rd
);
weight_L07O00 u0 (
    .q(rd[0*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O01 u1 (
    .q(rd[1*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O02 u2 (
    .q(rd[2*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O03 u3 (
    .q(rd[3*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O04 u4 (
    .q(rd[4*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O05 u5 (
    .q(rd[5*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O06 u6 (
    .q(rd[6*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O07 u7 (
    .q(rd[7*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O08 u8 (
    .q(rd[8*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O09 u9 (
    .q(rd[9*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O10 u10 (
    .q(rd[10*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O11 u11 (
    .q(rd[11*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O12 u12 (
    .q(rd[12*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O13 u13 (
    .q(rd[13*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O14 u14 (
    .q(rd[14*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O15 u15 (
    .q(rd[15*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O16 u16 (
    .q(rd[16*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O17 u17 (
    .q(rd[17*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O18 u18 (
    .q(rd[18*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O19 u19 (
    .q(rd[19*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O20 u20 (
    .q(rd[20*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O21 u21 (
    .q(rd[21*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O22 u22 (
    .q(rd[22*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O23 u23 (
    .q(rd[23*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O24 u24 (
    .q(rd[24*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O25 u25 (
    .q(rd[25*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O26 u26 (
    .q(rd[26*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O27 u27 (
    .q(rd[27*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O28 u28 (
    .q(rd[28*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O29 u29 (
    .q(rd[29*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O30 u30 (
    .q(rd[30*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
weight_L07O31 u31 (
    .q(rd[31*18*8 +:18*8]),
    .address(ra),
    .clock(clk)
);
endmodule

module LAYER8 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(8):0] s_chan,
    input wire s_last,
    input wire [$clog2(14):0] s_col,
    input wire [$clog2(14):0] s_row,
    input wire [32*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(4):0] m_chan,
    output reg m_last,
    output reg [$clog2(14):0] m_col,
    output reg [$clog2(14):0] m_row,
    output reg [64*32-1:0] m_data
);

parameter OCMUX=4,OCHAN=256,OWIDTH=14,TDMPAD=   37027;

reg [$clog2(896):0] strip_wa [1-1:0];
reg strip_wen [1-1:0];
reg signed [$clog2(896)+1:0] strip_ra [1-1:0];
reg signed [$clog2(2304)+1:0] weight_ra;
reg [1-1:0] strip_zpad;
reg [1-1:0] strip_zpad_q;
reg signed [$clog2(256)+1:0] ic;
reg [$clog2(256):0] ic_q,ic_qq;
reg [$clog2(4):0] ochan_sel;
reg [$clog2(1):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(8):0] s_chan_q;
reg [$clog2(14):0] s_col_q;
reg [$clog2(14):0] s_row_q;
reg [32*32-1:0] s_data_q;
reg [32*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%7)*(14+2)*8+(s_col_q-(0*14-1))*8+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*14) && (s_col_q < (0+1)*14+2-1);
end
wire [32*32-1:0] strip_rd [1-1:0];
reg [32-1:0] patch [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : STRIP
STRIP #(896,1024) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(896):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(256):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/8)*32 +:32];
    end
end
endgenerate

wire [18*256-1:0] weight_rd;
wire [32*256-1:0] bias_rd;

WEIGHT8 weight (.clk(m_clk),.ra(weight_ra[$clog2(2304):0]),.rd(weight_rd));
BIAS8 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [64-1:0];
reg [32-1:0] bias_mux [64-1:0];
generate
for (i=0; i<64; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*4+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*4+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [64*32-1:0] feat [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : ALU_NS
    for (j=0; j<64; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(14):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==14-1) begin
            icount <= 'd0;
            if (s_row_q >= 1)
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(3)+1:0] ky;
reg signed [$clog2(3)+1:0] kx;
reg signed [$clog2(14)+1:0] ocol;
reg signed [$clog2(14)+1:0] orow;
reg [$clog2(37027):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= ((orow==0)&&(ky<0)) || ((ocol==0)&&(kx<0)) || ((orow==13)&&(ky>0)) || ((ocol==13)&&(kx>0));
strip_ra[0] <= ((ky+(orow*1))%7)*(14+2)*8+kx*8+ocol*1*8+(ic%8)+8;
weight_ra <= (ky+1)*3*256+(kx+1)*256+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==255) begin
                if (kx==1) begin
                    if (ky==1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd6;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd6;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd6;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd6;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<1) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*14+ocol < 14) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*14+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==4-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==4-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==14-1) begin
                        ocol <= 'd0;
                        if (orow==14-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           if (orow==14-1) begin
               ocol <= 'd0;
               ochan_sel <= 'd0;
               alu_op <= 'd0;
               m_rowwait_count <= 0;
               m_state<= 'd10;
           end
           else begin
               m_row_ack <= 1'b1;
               if (~m_row_req) begin
                   m_row_ack <= 1'b0;
                   m_state<= 'd0;
               end
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS8 (
    input wire clk,
    output wire [256*32-1:0] rd
);
bias_L08 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT8 (
    input wire clk,
    input wire [$clog2(2304):0] ra,
    output wire [256*18-1:0] rd
);
weight_L08O00 u0 (
    .q(rd[0*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O01 u1 (
    .q(rd[1*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O02 u2 (
    .q(rd[2*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O03 u3 (
    .q(rd[3*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O04 u4 (
    .q(rd[4*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O05 u5 (
    .q(rd[5*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O06 u6 (
    .q(rd[6*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O07 u7 (
    .q(rd[7*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O08 u8 (
    .q(rd[8*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O09 u9 (
    .q(rd[9*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O10 u10 (
    .q(rd[10*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O11 u11 (
    .q(rd[11*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O12 u12 (
    .q(rd[12*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O13 u13 (
    .q(rd[13*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O14 u14 (
    .q(rd[14*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O15 u15 (
    .q(rd[15*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O16 u16 (
    .q(rd[16*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O17 u17 (
    .q(rd[17*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O18 u18 (
    .q(rd[18*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O19 u19 (
    .q(rd[19*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O20 u20 (
    .q(rd[20*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O21 u21 (
    .q(rd[21*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O22 u22 (
    .q(rd[22*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O23 u23 (
    .q(rd[23*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O24 u24 (
    .q(rd[24*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O25 u25 (
    .q(rd[25*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O26 u26 (
    .q(rd[26*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O27 u27 (
    .q(rd[27*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O28 u28 (
    .q(rd[28*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O29 u29 (
    .q(rd[29*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O30 u30 (
    .q(rd[30*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O31 u31 (
    .q(rd[31*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O32 u32 (
    .q(rd[32*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O33 u33 (
    .q(rd[33*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O34 u34 (
    .q(rd[34*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O35 u35 (
    .q(rd[35*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O36 u36 (
    .q(rd[36*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O37 u37 (
    .q(rd[37*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O38 u38 (
    .q(rd[38*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O39 u39 (
    .q(rd[39*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O40 u40 (
    .q(rd[40*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O41 u41 (
    .q(rd[41*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O42 u42 (
    .q(rd[42*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O43 u43 (
    .q(rd[43*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O44 u44 (
    .q(rd[44*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O45 u45 (
    .q(rd[45*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O46 u46 (
    .q(rd[46*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O47 u47 (
    .q(rd[47*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O48 u48 (
    .q(rd[48*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O49 u49 (
    .q(rd[49*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O50 u50 (
    .q(rd[50*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O51 u51 (
    .q(rd[51*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O52 u52 (
    .q(rd[52*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O53 u53 (
    .q(rd[53*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O54 u54 (
    .q(rd[54*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O55 u55 (
    .q(rd[55*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O56 u56 (
    .q(rd[56*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O57 u57 (
    .q(rd[57*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O58 u58 (
    .q(rd[58*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O59 u59 (
    .q(rd[59*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O60 u60 (
    .q(rd[60*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O61 u61 (
    .q(rd[61*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O62 u62 (
    .q(rd[62*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L08O63 u63 (
    .q(rd[63*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
endmodule

module LAYER9 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(4):0] s_chan,
    input wire s_last,
    input wire [$clog2(14):0] s_col,
    input wire [$clog2(14):0] s_row,
    input wire [64*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(4):0] m_chan,
    output reg m_last,
    output reg [$clog2(14):0] m_col,
    output reg [$clog2(14):0] m_row,
    output reg [64*32-1:0] m_data
);

parameter OCMUX=4,OCHAN=256,OWIDTH=14,TDMPAD=   37027;

reg [$clog2(448):0] strip_wa [1-1:0];
reg strip_wen [1-1:0];
reg signed [$clog2(448)+1:0] strip_ra [1-1:0];
reg signed [$clog2(2304)+1:0] weight_ra;
reg [1-1:0] strip_zpad;
reg [1-1:0] strip_zpad_q;
reg signed [$clog2(256)+1:0] ic;
reg [$clog2(256):0] ic_q,ic_qq;
reg [$clog2(4):0] ochan_sel;
reg [$clog2(1):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(4):0] s_chan_q;
reg [$clog2(14):0] s_col_q;
reg [$clog2(14):0] s_row_q;
reg [64*32-1:0] s_data_q;
reg [64*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%7)*(14+2)*4+(s_col_q-(0*14-1))*4+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*14) && (s_col_q < (0+1)*14+2-1);
end
wire [64*32-1:0] strip_rd [1-1:0];
reg [32-1:0] patch [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : STRIP
STRIP #(448,2048) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(448):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(256):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/4)*32 +:32];
    end
end
endgenerate

wire [18*256-1:0] weight_rd;
wire [32*256-1:0] bias_rd;

WEIGHT9 weight (.clk(m_clk),.ra(weight_ra[$clog2(2304):0]),.rd(weight_rd));
BIAS9 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [64-1:0];
reg [32-1:0] bias_mux [64-1:0];
generate
for (i=0; i<64; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*4+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*4+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [64*32-1:0] feat [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : ALU_NS
    for (j=0; j<64; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(14):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==14-1) begin
            icount <= 'd0;
            if (s_row_q >= 1)
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(3)+1:0] ky;
reg signed [$clog2(3)+1:0] kx;
reg signed [$clog2(14)+1:0] ocol;
reg signed [$clog2(14)+1:0] orow;
reg [$clog2(37027):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= ((orow==0)&&(ky<0)) || ((ocol==0)&&(kx<0)) || ((orow==13)&&(ky>0)) || ((ocol==13)&&(kx>0));
strip_ra[0] <= ((ky+(orow*1))%7)*(14+2)*4+kx*4+ocol*1*4+(ic%4)+4;
weight_ra <= (ky+1)*3*256+(kx+1)*256+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==255) begin
                if (kx==1) begin
                    if (ky==1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd6;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd6;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd6;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd6;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<1) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*14+ocol < 14) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*14+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==4-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==4-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==14-1) begin
                        ocol <= 'd0;
                        if (orow==14-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           if (orow==14-1) begin
               ocol <= 'd0;
               ochan_sel <= 'd0;
               alu_op <= 'd0;
               m_rowwait_count <= 0;
               m_state<= 'd10;
           end
           else begin
               m_row_ack <= 1'b1;
               if (~m_row_req) begin
                   m_row_ack <= 1'b0;
                   m_state<= 'd0;
               end
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS9 (
    input wire clk,
    output wire [256*32-1:0] rd
);
bias_L09 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT9 (
    input wire clk,
    input wire [$clog2(2304):0] ra,
    output wire [256*18-1:0] rd
);
weight_L09O00 u0 (
    .q(rd[0*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O01 u1 (
    .q(rd[1*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O02 u2 (
    .q(rd[2*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O03 u3 (
    .q(rd[3*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O04 u4 (
    .q(rd[4*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O05 u5 (
    .q(rd[5*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O06 u6 (
    .q(rd[6*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O07 u7 (
    .q(rd[7*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O08 u8 (
    .q(rd[8*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O09 u9 (
    .q(rd[9*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O10 u10 (
    .q(rd[10*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O11 u11 (
    .q(rd[11*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O12 u12 (
    .q(rd[12*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O13 u13 (
    .q(rd[13*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O14 u14 (
    .q(rd[14*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O15 u15 (
    .q(rd[15*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O16 u16 (
    .q(rd[16*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O17 u17 (
    .q(rd[17*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O18 u18 (
    .q(rd[18*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O19 u19 (
    .q(rd[19*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O20 u20 (
    .q(rd[20*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O21 u21 (
    .q(rd[21*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O22 u22 (
    .q(rd[22*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O23 u23 (
    .q(rd[23*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O24 u24 (
    .q(rd[24*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O25 u25 (
    .q(rd[25*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O26 u26 (
    .q(rd[26*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O27 u27 (
    .q(rd[27*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O28 u28 (
    .q(rd[28*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O29 u29 (
    .q(rd[29*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O30 u30 (
    .q(rd[30*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O31 u31 (
    .q(rd[31*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O32 u32 (
    .q(rd[32*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O33 u33 (
    .q(rd[33*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O34 u34 (
    .q(rd[34*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O35 u35 (
    .q(rd[35*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O36 u36 (
    .q(rd[36*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O37 u37 (
    .q(rd[37*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O38 u38 (
    .q(rd[38*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O39 u39 (
    .q(rd[39*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O40 u40 (
    .q(rd[40*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O41 u41 (
    .q(rd[41*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O42 u42 (
    .q(rd[42*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O43 u43 (
    .q(rd[43*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O44 u44 (
    .q(rd[44*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O45 u45 (
    .q(rd[45*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O46 u46 (
    .q(rd[46*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O47 u47 (
    .q(rd[47*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O48 u48 (
    .q(rd[48*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O49 u49 (
    .q(rd[49*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O50 u50 (
    .q(rd[50*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O51 u51 (
    .q(rd[51*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O52 u52 (
    .q(rd[52*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O53 u53 (
    .q(rd[53*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O54 u54 (
    .q(rd[54*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O55 u55 (
    .q(rd[55*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O56 u56 (
    .q(rd[56*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O57 u57 (
    .q(rd[57*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O58 u58 (
    .q(rd[58*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O59 u59 (
    .q(rd[59*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O60 u60 (
    .q(rd[60*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O61 u61 (
    .q(rd[61*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O62 u62 (
    .q(rd[62*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
weight_L09O63 u63 (
    .q(rd[63*18*4 +:18*4]),
    .address(ra),
    .clock(clk)
);
endmodule

module LAYER10 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(4):0] s_chan,
    input wire s_last,
    input wire [$clog2(14):0] s_col,
    input wire [$clog2(14):0] s_row,
    input wire [64*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(16):0] m_chan,
    output reg m_last,
    output reg [$clog2(7):0] m_col,
    output reg [$clog2(7):0] m_row,
    output reg [16*32-1:0] m_data
);

parameter OCMUX=16,OCHAN=256,OWIDTH=7,TDMPAD=   74053;

reg [$clog2(420):0] strip_wa [1-1:0];
reg strip_wen [1-1:0];
reg signed [$clog2(420)+1:0] strip_ra [1-1:0];
reg signed [$clog2(2304)+1:0] weight_ra;
reg [1-1:0] strip_zpad;
reg [1-1:0] strip_zpad_q;
reg signed [$clog2(256)+1:0] ic;
reg [$clog2(256):0] ic_q,ic_qq;
reg [$clog2(16):0] ochan_sel;
reg [$clog2(1):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(4):0] s_chan_q;
reg [$clog2(14):0] s_col_q;
reg [$clog2(14):0] s_row_q;
reg [64*32-1:0] s_data_q;
reg [64*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%7)*(14+1)*4+(s_col_q-(0*14-1))*4+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*14) && (s_col_q < (0+1)*14+1-1);
end
wire [64*32-1:0] strip_rd [1-1:0];
reg [32-1:0] patch [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : STRIP
STRIP #(420,2048) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(420):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(256):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/4)*32 +:32];
    end
end
endgenerate

wire [18*256-1:0] weight_rd;
wire [32*256-1:0] bias_rd;

WEIGHT10 weight (.clk(m_clk),.ra(weight_ra[$clog2(2304):0]),.rd(weight_rd));
BIAS10 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [16-1:0];
reg [32-1:0] bias_mux [16-1:0];
generate
for (i=0; i<16; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*16+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*16+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [16*32-1:0] feat [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : ALU_NS
    for (j=0; j<16; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(14):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==14-1) begin
            icount <= 'd0;
            if ((s_row_q >= 1) && ((s_row_q%2) == (3%2)))
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(3)+1:0] ky;
reg signed [$clog2(3)+1:0] kx;
reg signed [$clog2(7)+1:0] ocol;
reg signed [$clog2(7)+1:0] orow;
reg [$clog2(74054):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= ((orow==0)&&(ky<0)) || ((ocol==0)&&(kx<0));
strip_ra[0] <= ((ky+(orow*2))%7)*(14+1)*4+kx*4+ocol*2*4+(ic%4)+4;
weight_ra <= (ky+1)*3*256+(kx+1)*256+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==255) begin
                if (kx==1) begin
                    if (ky==1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd6;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd6;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd6;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd6;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<1) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*7+ocol < 7) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*7+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==16-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==16-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==7-1) begin
                        ocol <= 'd0;
                        if (orow==7-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           m_row_ack <= 1'b1;
           if (~m_row_req) begin
               m_row_ack <= 1'b0;
               m_state<= 'd0;
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS10 (
    input wire clk,
    output wire [256*32-1:0] rd
);
bias_L10 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT10 (
    input wire clk,
    input wire [$clog2(2304):0] ra,
    output wire [256*18-1:0] rd
);
weight_L10O00 u0 (
    .q(rd[0*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O01 u1 (
    .q(rd[1*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O02 u2 (
    .q(rd[2*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O03 u3 (
    .q(rd[3*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O04 u4 (
    .q(rd[4*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O05 u5 (
    .q(rd[5*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O06 u6 (
    .q(rd[6*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O07 u7 (
    .q(rd[7*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O08 u8 (
    .q(rd[8*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O09 u9 (
    .q(rd[9*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O10 u10 (
    .q(rd[10*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O11 u11 (
    .q(rd[11*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O12 u12 (
    .q(rd[12*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O13 u13 (
    .q(rd[13*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O14 u14 (
    .q(rd[14*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L10O15 u15 (
    .q(rd[15*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
endmodule

module LAYER11 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(16):0] s_chan,
    input wire s_last,
    input wire [$clog2(7):0] s_col,
    input wire [$clog2(7):0] s_row,
    input wire [16*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(16):0] m_chan,
    output reg m_last,
    output reg [$clog2(5):0] m_col,
    output reg [$clog2(5):0] m_row,
    output reg [16*32-1:0] m_data
);

parameter OCMUX=16,OCHAN=256,OWIDTH=5,TDMPAD=  148133;

reg [$clog2(784):0] strip_wa [1-1:0];
reg strip_wen [1-1:0];
reg signed [$clog2(784)+1:0] strip_ra [1-1:0];
reg signed [$clog2(2304)+1:0] weight_ra;
reg [1-1:0] strip_zpad;
reg [1-1:0] strip_zpad_q;
reg signed [$clog2(256)+1:0] ic;
reg [$clog2(256):0] ic_q,ic_qq;
reg [$clog2(16):0] ochan_sel;
reg [$clog2(1):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(16):0] s_chan_q;
reg [$clog2(7):0] s_col_q;
reg [$clog2(7):0] s_row_q;
reg [16*32-1:0] s_data_q;
reg [16*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%7)*(5+2)*16+(s_col_q-(0*5-1))*16+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*5) && (s_col_q < (0+1)*5+2-1);
end
wire [16*32-1:0] strip_rd [1-1:0];
reg [32-1:0] patch [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : STRIP
STRIP #(784,512) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(784):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(256):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/16)*32 +:32];
    end
end
endgenerate

wire [18*256-1:0] weight_rd;
wire [32*256-1:0] bias_rd;

WEIGHT11 weight (.clk(m_clk),.ra(weight_ra[$clog2(2304):0]),.rd(weight_rd));
BIAS11 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [16-1:0];
reg [32-1:0] bias_mux [16-1:0];
generate
for (i=0; i<16; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*16+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*16+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [16*32-1:0] feat [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : ALU_NS
    for (j=0; j<16; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(7):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==7-1) begin
            icount <= 'd0;
            if (s_row_q >= 2)
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(3)+1:0] ky;
reg signed [$clog2(3)+1:0] kx;
reg signed [$clog2(5)+1:0] ocol;
reg signed [$clog2(5)+1:0] orow;
reg [$clog2(148134):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= 'b0;
strip_ra[0] <= (((ky+1)+(orow*1))%7)*(5+2)*16+(kx+1)*16+ocol*1*16+(ic%16)+16;
weight_ra <= (ky+1)*3*256+(kx+1)*256+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==255) begin
                if (kx==1) begin
                    if (ky==1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd6;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd6;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd6;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd6;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<1) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*5+ocol < 5) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*5+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==16-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==16-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==5-1) begin
                        ocol <= 'd0;
                        if (orow==5-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           if (orow==5-1) begin
               ocol <= 'd0;
               ochan_sel <= 'd0;
               alu_op <= 'd0;
               m_rowwait_count <= 0;
               m_state<= 'd10;
           end
           else begin
               m_row_ack <= 1'b1;
               if (~m_row_req) begin
                   m_row_ack <= 1'b0;
                   m_state<= 'd0;
               end
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS11 (
    input wire clk,
    output wire [256*32-1:0] rd
);
bias_L11 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT11 (
    input wire clk,
    input wire [$clog2(2304):0] ra,
    output wire [256*18-1:0] rd
);
weight_L11O00 u0 (
    .q(rd[0*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O01 u1 (
    .q(rd[1*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O02 u2 (
    .q(rd[2*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O03 u3 (
    .q(rd[3*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O04 u4 (
    .q(rd[4*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O05 u5 (
    .q(rd[5*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O06 u6 (
    .q(rd[6*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O07 u7 (
    .q(rd[7*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O08 u8 (
    .q(rd[8*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O09 u9 (
    .q(rd[9*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O10 u10 (
    .q(rd[10*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O11 u11 (
    .q(rd[11*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O12 u12 (
    .q(rd[12*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O13 u13 (
    .q(rd[13*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O14 u14 (
    .q(rd[14*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
weight_L11O15 u15 (
    .q(rd[15*18*16 +:18*16]),
    .address(ra),
    .clock(clk)
);
endmodule

module LAYER12 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(16):0] s_chan,
    input wire s_last,
    input wire [$clog2(5):0] s_col,
    input wire [$clog2(5):0] s_row,
    input wire [16*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(32):0] m_chan,
    output reg m_last,
    output reg [$clog2(3):0] m_col,
    output reg [$clog2(3):0] m_row,
    output reg [8*32-1:0] m_data
);

parameter OCMUX=32,OCHAN=256,OWIDTH=3,TDMPAD=  111093;

reg [$clog2(560):0] strip_wa [1-1:0];
reg strip_wen [1-1:0];
reg signed [$clog2(560)+1:0] strip_ra [1-1:0];
reg signed [$clog2(2304)+1:0] weight_ra;
reg [1-1:0] strip_zpad;
reg [1-1:0] strip_zpad_q;
reg signed [$clog2(256)+1:0] ic;
reg [$clog2(256):0] ic_q,ic_qq;
reg [$clog2(32):0] ochan_sel;
reg [$clog2(1):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(16):0] s_chan_q;
reg [$clog2(5):0] s_col_q;
reg [$clog2(5):0] s_row_q;
reg [16*32-1:0] s_data_q;
reg [16*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%7)*(3+2)*16+(s_col_q-(0*3-1))*16+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*3) && (s_col_q < (0+1)*3+2-1);
end
wire [16*32-1:0] strip_rd [1-1:0];
reg [32-1:0] patch [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : STRIP
STRIP #(560,512) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(560):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(256):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/16)*32 +:32];
    end
end
endgenerate

wire [18*256-1:0] weight_rd;
wire [32*256-1:0] bias_rd;

WEIGHT12 weight (.clk(m_clk),.ra(weight_ra[$clog2(2304):0]),.rd(weight_rd));
BIAS12 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [8-1:0];
reg [32-1:0] bias_mux [8-1:0];
generate
for (i=0; i<8; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*32+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*32+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [8*32-1:0] feat [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : ALU_NS
    for (j=0; j<8; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(5):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==5-1) begin
            icount <= 'd0;
            if (s_row_q >= 2)
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(3)+1:0] ky;
reg signed [$clog2(3)+1:0] kx;
reg signed [$clog2(3)+1:0] ocol;
reg signed [$clog2(3)+1:0] orow;
reg [$clog2(111094):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= 'b0;
strip_ra[0] <= (((ky+1)+(orow*1))%7)*(3+2)*16+(kx+1)*16+ocol*1*16+(ic%16)+16;
weight_ra <= (ky+1)*3*256+(kx+1)*256+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==255) begin
                if (kx==1) begin
                    if (ky==1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd6;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd6;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd6;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd6;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<1) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*3+ocol < 3) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*3+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==32-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==32-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==3-1) begin
                        ocol <= 'd0;
                        if (orow==3-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           if (orow==3-1) begin
               ocol <= 'd0;
               ochan_sel <= 'd0;
               alu_op <= 'd0;
               m_rowwait_count <= 0;
               m_state<= 'd10;
           end
           else begin
               m_row_ack <= 1'b1;
               if (~m_row_req) begin
                   m_row_ack <= 1'b0;
                   m_state<= 'd0;
               end
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS12 (
    input wire clk,
    output wire [256*32-1:0] rd
);
bias_L12 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT12 (
    input wire clk,
    input wire [$clog2(2304):0] ra,
    output wire [256*18-1:0] rd
);
weight_L12O00 u0 (
    .q(rd[0*18*32 +:18*32]),
    .address(ra),
    .clock(clk)
);
weight_L12O01 u1 (
    .q(rd[1*18*32 +:18*32]),
    .address(ra),
    .clock(clk)
);
weight_L12O02 u2 (
    .q(rd[2*18*32 +:18*32]),
    .address(ra),
    .clock(clk)
);
weight_L12O03 u3 (
    .q(rd[3*18*32 +:18*32]),
    .address(ra),
    .clock(clk)
);
weight_L12O04 u4 (
    .q(rd[4*18*32 +:18*32]),
    .address(ra),
    .clock(clk)
);
weight_L12O05 u5 (
    .q(rd[5*18*32 +:18*32]),
    .address(ra),
    .clock(clk)
);
weight_L12O06 u6 (
    .q(rd[6*18*32 +:18*32]),
    .address(ra),
    .clock(clk)
);
weight_L12O07 u7 (
    .q(rd[7*18*32 +:18*32]),
    .address(ra),
    .clock(clk)
);
endmodule

module LAYER13 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(32):0] s_chan,
    input wire s_last,
    input wire [$clog2(3):0] s_col,
    input wire [$clog2(3):0] s_row,
    input wire [8*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(64):0] m_chan,
    output reg m_last,
    output reg [$clog2(1):0] m_col,
    output reg [$clog2(1):0] m_row,
    output reg [4*32-1:0] m_data
);

parameter OCMUX=64,OCHAN=256,OWIDTH=1,TDMPAD=  185173;

reg [$clog2(672):0] strip_wa [1-1:0];
reg strip_wen [1-1:0];
reg signed [$clog2(672)+1:0] strip_ra [1-1:0];
reg signed [$clog2(2304)+1:0] weight_ra;
reg [1-1:0] strip_zpad;
reg [1-1:0] strip_zpad_q;
reg signed [$clog2(256)+1:0] ic;
reg [$clog2(256):0] ic_q,ic_qq;
reg [$clog2(64):0] ochan_sel;
reg [$clog2(1):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(32):0] s_chan_q;
reg [$clog2(3):0] s_col_q;
reg [$clog2(3):0] s_row_q;
reg [8*32-1:0] s_data_q;
reg [8*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%7)*(1+2)*32+(s_col_q-(0*1-1))*32+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*1) && (s_col_q < (0+1)*1+2-1);
end
wire [8*32-1:0] strip_rd [1-1:0];
reg [32-1:0] patch [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : STRIP
STRIP #(672,256) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(672):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(256):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/32)*32 +:32];
    end
end
endgenerate

wire [18*256-1:0] weight_rd;
wire [32*256-1:0] bias_rd;

WEIGHT13 weight (.clk(m_clk),.ra(weight_ra[$clog2(2304):0]),.rd(weight_rd));
BIAS13 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [4-1:0];
reg [32-1:0] bias_mux [4-1:0];
generate
for (i=0; i<4; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*64+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*64+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [4*32-1:0] feat [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : ALU_NS
    for (j=0; j<4; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(3):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==3-1) begin
            icount <= 'd0;
            if (s_row_q >= 2)
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(3)+1:0] ky;
reg signed [$clog2(3)+1:0] kx;
reg signed [$clog2(1)+1:0] ocol;
reg signed [$clog2(1)+1:0] orow;
reg [$clog2(185174):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= 'b0;
strip_ra[0] <= (((ky+1)+(orow*1))%7)*(1+2)*32+(kx+1)*32+ocol*1*32+(ic%32)+32;
weight_ra <= (ky+1)*3*256+(kx+1)*256+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==255) begin
                if (kx==1) begin
                    if (ky==1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd6;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd6;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd6;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd6;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<1) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*1+ocol < 1) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*1+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==64-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==64-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==1-1) begin
                        ocol <= 'd0;
                        if (orow==1-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           if (orow==1-1) begin
               ocol <= 'd0;
               ochan_sel <= 'd0;
               alu_op <= 'd0;
               m_rowwait_count <= 0;
               m_state<= 'd10;
           end
           else begin
               m_row_ack <= 1'b1;
               if (~m_row_req) begin
                   m_row_ack <= 1'b0;
                   m_state<= 'd0;
               end
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS13 (
    input wire clk,
    output wire [256*32-1:0] rd
);
bias_L13 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT13 (
    input wire clk,
    input wire [$clog2(2304):0] ra,
    output wire [256*18-1:0] rd
);
weight_L13O00 u0 (
    .q(rd[0*18*64 +:18*64]),
    .address(ra),
    .clock(clk)
);
weight_L13O01 u1 (
    .q(rd[1*18*64 +:18*64]),
    .address(ra),
    .clock(clk)
);
weight_L13O02 u2 (
    .q(rd[2*18*64 +:18*64]),
    .address(ra),
    .clock(clk)
);
weight_L13O03 u3 (
    .q(rd[3*18*64 +:18*64]),
    .address(ra),
    .clock(clk)
);
endmodule

module LAYER14 (
    input wire s_clk,
    input wire m_clk,
    input wire reset,
    input wire s_valid,
    input wire [$clog2(64):0] s_chan,
    input wire s_last,
    input wire [$clog2(1):0] s_col,
    input wire [$clog2(1):0] s_row,
    input wire [4*32-1:0] s_data,
    output reg m_valid,
    output reg [$clog2(64):0] m_chan,
    output reg m_last,
    output reg [$clog2(1):0] m_col,
    output reg [$clog2(1):0] m_row,
    output reg [8*32-1:0] m_data
);

parameter OCMUX=64,OCHAN=512,OWIDTH=1,TDMPAD=  316245;

reg [$clog2(960):0] strip_wa [1-1:0];
reg strip_wen [1-1:0];
reg signed [$clog2(960)+1:0] strip_ra [1-1:0];
reg signed [$clog2(256)+1:0] weight_ra;
reg [1-1:0] strip_zpad;
reg [1-1:0] strip_zpad_q;
reg signed [$clog2(256)+1:0] ic;
reg [$clog2(256):0] ic_q,ic_qq;
reg [$clog2(64):0] ochan_sel;
reg [$clog2(1):0] strip_sel;
reg [4:0] alu_op;

genvar i,j;
reg s_valid_q;
reg s_last_q;
reg [$clog2(64):0] s_chan_q;
reg [$clog2(1):0] s_col_q;
reg [$clog2(1):0] s_row_q;
reg [4*32-1:0] s_data_q;
reg [4*32-1:0] s_data_qq;
always @(posedge s_clk) begin
    s_valid_q <= s_valid;
    s_chan_q <= s_chan;
    s_last_q <= s_last;
    s_col_q <= s_col;
    s_row_q <= s_row;
    s_data_q <= s_data;
    s_data_qq <= s_data_q;
end
always @ (posedge s_clk) begin
    strip_wa[0] <= (s_row_q%5)*(1+2)*64+(s_col_q-(0*1-1))*64+s_chan_q;
    strip_wen[0] <= s_valid_q && (s_col_q >= 0*1) && (s_col_q < (0+1)*1+2-1);
end
wire [4*32-1:0] strip_rd [1-1:0];
reg [32-1:0] patch [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : STRIP
STRIP #(960,128) strip (.wclk(s_clk),.rclk(m_clk),.reset(reset),.ra(strip_ra[i][$clog2(960):0]),.rd(strip_rd[i]),.wen(strip_wen[i]),.wa(strip_wa[i]),.wd(s_data_qq));
    always @(posedge m_clk) begin
        ic_q <= ic[$clog2(256):0];
        ic_qq <= ic_q;
        patch[i] <= strip_zpad_q[i] ? 'd0 : strip_rd[i][(ic_qq/64)*32 +:32];
    end
end
endgenerate

wire [18*512-1:0] weight_rd;
wire [32*512-1:0] bias_rd;

WEIGHT14 weight (.clk(m_clk),.ra(weight_ra[$clog2(256):0]),.rd(weight_rd));
BIAS14 bias (.clk(m_clk),.rd(bias_rd));

reg [18-1:0] weight_mux [8-1:0];
reg [32-1:0] bias_mux [8-1:0];
generate
for (i=0; i<8; i=i+1) begin : WEIGHT_MUX
    always @(posedge m_clk) begin
        weight_mux[i] <= weight_rd[(i*64+ochan_sel)*18 +:18];
        bias_mux[i] <= bias_rd[(i*64+ochan_sel)*32 +:32];
    end
end
endgenerate

wire [8*32-1:0] feat [1-1:0];
generate
for (i=0; i<1; i=i+1) begin : ALU_NS
    for (j=0; j<8; j=j+1) begin : ALU_OC
        ALU alu (
        .clk(m_clk),
        .reset(reset),
        .alu_op(alu_op),
        .patch(patch[i]),
        .weight(weight_mux[j]),
        .bias(bias_mux[j]),
        .feat(feat[i][32*j +:32]));
    end
end
endgenerate

always @(posedge m_clk) begin
    m_data <= feat[strip_sel];
end


reg [$clog2(1):0] icount;
reg row_cyc;
always @ (posedge s_clk) begin : S_ICOUNT
    if (reset) begin
        icount <= 'd0;
        row_cyc <= 'b0;
    end
    else if (s_last_q) begin
        if (icount==1-1) begin
            icount <= 'd0;
            if (s_row_q >= 0)
                row_cyc <= 1'b1;
            else
                row_cyc <= 1'b0;
        end
        else begin
            icount <= icount+'d1;
            row_cyc <= 1'b0;
        end
    end
    else begin
        row_cyc <= 1'b0;
    end
end

reg s_row_req;
reg m_row_ack;
reg m_row_req0,m_row_req1,m_row_req;
reg s_row_ack0,s_row_ack1,s_row_ack;
always @(posedge m_clk) begin : M_SYNC
    m_row_req0 <= s_row_req;
    m_row_req1 <= m_row_req0;
    m_row_req <= m_row_req1;
end
always @(posedge s_clk) begin : S_SYNC
    s_row_ack0 <= m_row_ack;
    s_row_ack1 <= s_row_ack0;
    s_row_ack <= s_row_ack1;
end

reg [1:0] s_state;
reg row_go, row_clr;
always @ (posedge s_clk) begin : S_FSM
    if (reset | row_clr)
        row_go <= 1'b0;
    else if (row_cyc)
        row_go <= 1'b1;
    if (reset) begin
        s_state <= 'd0;
        row_clr <= 1'b0;
    end
    else if (s_state=='d0) begin
        if (row_go) begin
            s_state <= 'd1;
            row_clr <= 1'b1;
        end
    end
    else if (s_state=='d1) begin
        row_clr <= 1'b0;
        s_row_req <= 1'b1;
        if (s_row_ack) begin
            s_state <= 'd2;
        end
    end
    else if (s_state=='d2) begin
        s_row_req <= 1'b0;
        if (~s_row_ack) begin
            s_state <= 'd0;
        end
    end
end

reg [4:0] m_state;
reg [3:0] pipe_count;
reg signed [$clog2(1)+1:0] ky;
reg signed [$clog2(1)+1:0] kx;
reg signed [$clog2(1)+1:0] ocol;
reg signed [$clog2(1)+1:0] orow;
reg [$clog2(316246):0] m_rowwait_count;

always @ (posedge m_clk) begin
strip_zpad_q[0] <= strip_zpad[0];
strip_zpad[0] <= 'b0;
strip_ra[0] <= (((ky+1)+(orow*1))%5)*(1+2)*64+(kx+1)*64+ocol*1*64+(ic%64)+64;
weight_ra <= (ky+1)*1*256+(kx+1)*256+ic;
end

always @(posedge m_clk) begin
    if (reset) begin
        m_state <= 'd0;
        orow <= 'd0;
        m_row_ack <= 1'b0;
        m_valid <= 1'b0;
        m_last <= 1'b0;
    end
    else begin
        case (m_state)
        'd0: begin
            ocol <= 'd0;
            ochan_sel <= 'd0;
            alu_op <= 'd0;
            if (m_row_req) begin
                m_state <= 'd1;
            end
        end
        'd1: begin
            ky <= -1;
            kx <= -1;
            ic <= 0;
            pipe_count <= 'd0;
            alu_op <= 'd1;
            m_state <= 'd2;
        end
        'd2: begin
            alu_op <= 'd2;
            if (ic==255) begin
                if (kx==-1) begin
                    if (ky==-1) begin
                        m_state <= 'd3;
                    end
                    else begin
                        ky <= ky+'d1;
                        kx <= -1;
                        ic <= 'd0;
                    end
                end
                else begin
                    kx <= kx+'d1;
                    ic <= 'd0;
                end
            end
            else begin
                ic <= ic+'d1;
            end
        end
        'd3: begin
            if (pipe_count==2) begin
                alu_op <= 'd4;
                m_state <= 'd4;
            end
            else
                pipe_count <= pipe_count+'d1;
        end
        'd4: begin
            alu_op <= 'd5;
            m_state <= 'd8;
            strip_sel <= 'd0;
        end
        'd8: begin
            alu_op <= 'd5;
            m_state <= 'd9;
            strip_sel <= 'd0;
        end
        'd9: begin
            alu_op <= 'd5;
            m_state <= 'd5;
            strip_sel <= 'd0;
        end
        'd5: begin
            alu_op <= 'd5;
            m_state <= 'd6;
        end
        'd6: begin
            alu_op <= 'd0;
            if (strip_sel<1) begin
                strip_sel <= strip_sel+'d1;
                if (strip_sel*1+ocol < 1) begin
                    m_valid <= 1'b1;
                    m_chan <= ochan_sel;
                    m_col <= strip_sel*1+ocol;
                    m_row <= orow;
                    m_last <= (ochan_sel==64-1);
                end
                else begin
                    m_valid <= 1'b0;
                    m_last <= 1'b0;
                end
            end
            else begin
                m_valid <= 1'b0;
                m_last <= 1'b0;
                if (ochan_sel==64-1) begin
                    ochan_sel <= 'd0;
                    if (ocol==1-1) begin
                        ocol <= 'd0;
                        if (orow==1-1)
                            orow <= 'd0;
                        else
                            orow <= orow+'d1;
                        m_state <= 'd7;
                    end
                    else begin
                        ocol <= ocol+'d1;
                        m_state <= 'd1;
                    end
                end
                else begin
                        m_state <= 'd1;
                    ochan_sel <= ochan_sel+'d1;
                end
            end
       end
       'd7: begin
           m_row_ack <= 1'b1;
           if (~m_row_req) begin
               m_row_ack <= 1'b0;
               m_state<= 'd0;
           end
       end
       'd10: begin
           if (m_rowwait_count==TDMPAD)
                m_state <= 'd1;
           else
               m_rowwait_count <= m_rowwait_count+1;
       end
       endcase
   end
end
endmodule

module BIAS14 (
    input wire clk,
    output wire [512*32-1:0] rd
);
bias_L14 u0 (
    .q(rd),
    .address('d0),
    .clock(clk)
);
endmodule

module WEIGHT14 (
    input wire clk,
    input wire [$clog2(256):0] ra,
    output wire [512*18-1:0] rd
);
weight_L14O00 u0 (
    .q(rd[0*18*64 +:18*64]),
    .address(ra),
    .clock(clk)
);
weight_L14O01 u1 (
    .q(rd[1*18*64 +:18*64]),
    .address(ra),
    .clock(clk)
);
weight_L14O02 u2 (
    .q(rd[2*18*64 +:18*64]),
    .address(ra),
    .clock(clk)
);
weight_L14O03 u3 (
    .q(rd[3*18*64 +:18*64]),
    .address(ra),
    .clock(clk)
);
weight_L14O04 u4 (
    .q(rd[4*18*64 +:18*64]),
    .address(ra),
    .clock(clk)
);
weight_L14O05 u5 (
    .q(rd[5*18*64 +:18*64]),
    .address(ra),
    .clock(clk)
);
weight_L14O06 u6 (
    .q(rd[6*18*64 +:18*64]),
    .address(ra),
    .clock(clk)
);
weight_L14O07 u7 (
    .q(rd[7*18*64 +:18*64]),
    .address(ra),
    .clock(clk)
);
endmodule

module STRIP #(parameter SDEPTH=0,SWIDTH=0) (
    input wire wclk,
    input wire rclk,
    input wire reset,
    input wire [$clog2(SDEPTH):0] ra,
    output reg [SWIDTH-1:0] rd,
    input wire wen,
    input wire [$clog2(SDEPTH):0] wa,
    input wire [SWIDTH-1:0] wd
);
(* ramstyle = "no_rw_check" *) reg [SWIDTH-1:0] dpram [0:SDEPTH-1];
always @(posedge rclk) begin
    rd <= dpram[ra];
end
always @(posedge wclk) begin
    if (wen) begin
        dpram[wa] <= wd;
    end
end
endmodule

module ALU (
    input wire clk,
    input wire reset,
    input wire [4:0] alu_op,
    input wire [32-1:0] patch,
    input wire [18-1:0] weight,
    input wire [32-1:0] bias,
    output wire [32-1:0] feat
);

wire acc;
wire [2:0] ena;
wire [32-1:0] mult_a;
wire [32-1:0] mult_b;
wire [32-1:0] result;
reg [32-1:0] reg_z;
mac_fp32 u0 (
    .accumulate  (acc),  //   input,   width = 1,  accumulate.accumulate
    .fp32_mult_a (mult_a), //   input,  width = 32, fp32_mult_a.fp32_mult_a
    .fp32_mult_b (mult_b), //   input,  width = 32, fp32_mult_b.fp32_mult_b
    .clk         (clk),         //   input,   width = 1,         clk.clk
    .ena         (ena),         //   input,   width = 3,         ena.ena
    .fp32_result (result)  //  output,  width = 32, fp32_result.fp32_result
);
assign mult_a = (alu_op=='d2)||(alu_op=='d1) ? patch : (alu_op=='d4) ? 32'h3f800000 : 32'h00000000;
assign mult_b = (alu_op=='d4) ? bias : {weight,14'b0};
assign ena = 3'b111;
assign acc = (alu_op=='d1) ? 1'b0 : 1'b1;
always @(posedge clk) begin
    if (alu_op=='d6)
        reg_z <= result[31] ? 32'b0 : result;
    else if (alu_op=='d5)
        reg_z <= result;
end
assign feat = reg_z;
endmodule


