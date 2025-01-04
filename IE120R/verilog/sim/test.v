//
// testbench for IE120R image encoder, corresponding .memh files are produced
// by test.py
//

`timescale 1ns / 1fs
module tb #(
    parameter DTYPE=32,     // width of activations
    parameter ROWTIME=10000,  // row time in ns
    parameter CLKPERIOD=4,   // clock period in ns
    parameter TOLERANCE=1e-9,
    parameter ROI=896
) ();

reg clk;
reg reset;
reg s_valid;
reg [$clog2(ROI):0] s_col;
reg [$clog2(ROI):0] s_row;
reg [3*DTYPE-1:0] s_data;
wire m_valid;
wire [$clog2(8):0] m_chan;
wire m_last;
wire [$clog2(7):0] m_col;
wire [$clog2(7):0] m_row;
wire [8*DTYPE-1:0] m_data;

integer i;
//debug
//initial #0 $dumpvars(2, tb);

// init
initial begin
    clk = 1'b0;
    forever begin
        #(CLKPERIOD/2) clk = ~clk;
    end
end

// input image
reg signed [3*DTYPE-1:0] input_data [0:ROI*ROI-1];
integer row,col,frame;

// layer activations
reg [16*DTYPE-1:0] activation1 [0:200704-1];
reg [16*DTYPE-1:0] activation2 [0:200704-1];
reg [16*DTYPE-1:0] activation3 [0:200704-1];
reg [32*DTYPE-1:0] activation4 [0:50176-1];
reg [32*DTYPE-1:0] activation5 [0:50176-1];
reg [32*DTYPE-1:0] activation6 [0:50176-1];
reg [96*DTYPE-1:0] activation7 [0:12544-1];
reg [96*DTYPE-1:0] activation8 [0:12544-1];
reg [96*DTYPE-1:0] activation9 [0:12544-1];
reg [96*DTYPE-1:0] activation10 [0:3136-1];
reg [96*DTYPE-1:0] activation11 [0:3136-1];
reg [96*DTYPE-1:0] activation12 [0:3136-1];
reg [192*DTYPE-1:0] activation13 [0:784-1];
reg [192*DTYPE-1:0] activation14 [0:784-1];
reg [192*DTYPE-1:0] activation15 [0:784-1];
reg [320*DTYPE-1:0] activation16 [0:196-1];
reg [320*DTYPE-1:0] activation17 [0:196-1];
reg [320*DTYPE-1:0] activation18 [0:196-1];
reg [512*DTYPE-1:0] activation19 [0:49-1];
reg [512*DTYPE-1:0] activation20 [0:49-1];
reg [512*DTYPE-1:0] activation21 [0:49-1];
reg [512*DTYPE-1:0] activation22 [0:49-1];

initial begin
    $readmemh("./memh/weight1.memh", tb.u0.layer1.weight.spram);
    $readmemh("./memh/weight2.memh", tb.u0.layer2.weight.spram);
    $readmemh("./memh/weight3.memh", tb.u0.layer3.weight.spram);
    $readmemh("./memh/weight4.memh", tb.u0.layer4.weight.spram);
    $readmemh("./memh/weight5.memh", tb.u0.layer5.weight.spram);
    $readmemh("./memh/weight6.memh", tb.u0.layer6.weight.spram);
    $readmemh("./memh/weight7.memh", tb.u0.layer7.weight.spram);
    $readmemh("./memh/weight8.memh", tb.u0.layer8.weight.spram);
    $readmemh("./memh/weight9.memh", tb.u0.layer9.weight.spram);
    $readmemh("./memh/weight10.memh", tb.u0.layer10.weight.spram);
    $readmemh("./memh/weight11.memh", tb.u0.layer11.weight.spram);
    $readmemh("./memh/weight12.memh", tb.u0.layer12.weight.spram);
    $readmemh("./memh/weight13.memh", tb.u0.layer13.weight.spram);
    $readmemh("./memh/weight14.memh", tb.u0.layer14.weight.spram);
    $readmemh("./memh/weight15.memh", tb.u0.layer15.weight.spram);
    $readmemh("./memh/weight16.memh", tb.u0.layer16.weight.spram);
    $readmemh("./memh/weight17.memh", tb.u0.layer17.weight.spram);
    $readmemh("./memh/weight18.memh", tb.u0.layer18.weight.spram);
    $readmemh("./memh/weight19.memh", tb.u0.layer19.weight.spram);
    $readmemh("./memh/weight20.memh", tb.u0.layer20.weight.spram);
    $readmemh("./memh/weight21.memh", tb.u0.layer21.weight.spram);
    $readmemh("./memh/weight22.memh", tb.u0.layer22.weight.spram);
    $display("Loaded weights");
    $readmemh("./memh/bias1.memh", tb.u0.layer1.bias.spram);
    $readmemh("./memh/bias2.memh", tb.u0.layer2.bias.spram);
    $readmemh("./memh/bias3.memh", tb.u0.layer3.bias.spram);
    $readmemh("./memh/bias4.memh", tb.u0.layer4.bias.spram);
    $readmemh("./memh/bias5.memh", tb.u0.layer5.bias.spram);
    $readmemh("./memh/bias6.memh", tb.u0.layer6.bias.spram);
    $readmemh("./memh/bias7.memh", tb.u0.layer7.bias.spram);
    $readmemh("./memh/bias8.memh", tb.u0.layer8.bias.spram);
    $readmemh("./memh/bias9.memh", tb.u0.layer9.bias.spram);
    $readmemh("./memh/bias10.memh", tb.u0.layer10.bias.spram);
    $readmemh("./memh/bias11.memh", tb.u0.layer11.bias.spram);
    $readmemh("./memh/bias12.memh", tb.u0.layer12.bias.spram);
    $readmemh("./memh/bias13.memh", tb.u0.layer13.bias.spram);
    $readmemh("./memh/bias14.memh", tb.u0.layer14.bias.spram);
    $readmemh("./memh/bias15.memh", tb.u0.layer15.bias.spram);
    $readmemh("./memh/bias16.memh", tb.u0.layer16.bias.spram);
    $readmemh("./memh/bias17.memh", tb.u0.layer17.bias.spram);
    $readmemh("./memh/bias18.memh", tb.u0.layer18.bias.spram);
    $readmemh("./memh/bias19.memh", tb.u0.layer19.bias.spram);
    $readmemh("./memh/bias20.memh", tb.u0.layer20.bias.spram);
    $readmemh("./memh/bias21.memh", tb.u0.layer21.bias.spram);
    $readmemh("./memh/bias22.memh", tb.u0.layer22.bias.spram);
    $display("Loaded bias");
    
    $readmemh("./memh/input.memh", input_data);
    $display("Loaded input image");
    
    $readmemh("./memh/activation1.memh", activation1);
    $readmemh("./memh/activation2.memh", activation2);
    $readmemh("./memh/activation3.memh", activation3);
    $readmemh("./memh/activation4.memh", activation4);
    $readmemh("./memh/activation5.memh", activation5);
    $readmemh("./memh/activation6.memh", activation6);
    $readmemh("./memh/activation7.memh", activation7);
    $readmemh("./memh/activation8.memh", activation8);
    $readmemh("./memh/activation9.memh", activation9);
    $readmemh("./memh/activation10.memh", activation10);
    $readmemh("./memh/activation11.memh", activation11);
    $readmemh("./memh/activation12.memh", activation12);
    $readmemh("./memh/activation13.memh", activation13);
    $readmemh("./memh/activation14.memh", activation14);
    $readmemh("./memh/activation15.memh", activation15);
    $readmemh("./memh/activation16.memh", activation16);
    $readmemh("./memh/activation17.memh", activation17);
    $readmemh("./memh/activation18.memh", activation18);
    $readmemh("./memh/activation19.memh", activation19);
    $readmemh("./memh/activation20.memh", activation20);
    $readmemh("./memh/activation21.memh", activation21);
    $readmemh("./memh/activation22.memh", activation22);
    $display("Loaded PyTorch layer activations");

    reset = 1'b1;
    s_valid = 1'b0;
    #500
    reset = 1'b0;
    #500
    $display("Deassert reset");

    for (frame=0; frame<3; frame=frame+1) begin
        for (row=0; row<ROI; row=row+1) begin
            for (col=0; col<ROI; col=col+1) begin
                @(negedge clk) begin
		    s_data <= input_data[row*ROI+col];
                    s_valid <= 1'b1;
                    s_col <= col;
                    s_row <= row;
                end
                @(posedge clk) begin
                    $display("INPUT time %f s_row %d s_col %d s_data %h",$realtime,s_row,s_col,s_data);
                    s_data <= 'bx;
                    s_valid <= 1'b0;
                end
            end
            #(ROWTIME-CLKPERIOD*ROI); // subtract transmit time
        end
    end
end

// output monitor
integer nfeat;
always @(negedge clk) begin
    if (reset)
        nfeat=0;
    else if (m_valid) begin
        $display("OUTPUT time %f nfeat %d / %d m_row %d m_col %d m_chan %d m_data %h",$realtime,nfeat,7*7*tb.u0.layer22.OCMUX,m_row,m_col,m_chan,m_data);
        nfeat=nfeat+1;
	if (nfeat==7*7*tb.u0.layer22.OCMUX) begin
	    $display("WAIT 3ms ...");
            #3000000 $finish();
	end
    end
end

//dut
ie120r u0 (
    .clk(clk),
    .reset(reset),
    .s_valid(s_valid),
    .s_col(s_col),
    .s_row(s_row),
    .s_data(s_data),
    .m_valid(m_valid),
    .m_chan(m_chan),
    .m_last(m_last),
    .m_col(m_col),
    .m_row(m_row),
    .m_data(m_data)
);
 
always @(negedge clk) begin
if (tb.u0.layer1.m_valid) begin
    $display("LAYER1 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer1.m_row,tb.u0.layer1.m_col,tb.u0.layer1.m_chan,tb.u0.layer1.m_data,tb.u0.layer1.OCMUX);
    for (i=0; i<tb.u0.layer1.OCHAN/tb.u0.layer1.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer1.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation1[tb.u0.layer1.m_row*tb.u0.layer1.OWIDTH+tb.u0.layer1.m_col][((tb.u0.layer1.OCHAN-1)-i*tb.u0.layer1.OCMUX-tb.u0.layer1.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer1.m_data[i*DTYPE +:DTYPE],activation1[tb.u0.layer1.m_row*tb.u0.layer1.OWIDTH+tb.u0.layer1.m_col][((tb.u0.layer1.OCHAN-1)-i*tb.u0.layer1.OCMUX-tb.u0.layer1.m_chan)*DTYPE +:DTYPE]);
        end
        //else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer1.m_data[i*DTYPE +:DTYPE],activation1[tb.u0.layer1.m_row*tb.u0.layer1.OWIDTH+tb.u0.layer1.m_col][((tb.u0.layer1.OCHAN-1)-i*tb.u0.layer1.OCMUX-tb.u0.layer1.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer2.m_valid) begin
    $display("LAYER2 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer2.m_row,tb.u0.layer2.m_col,tb.u0.layer2.m_chan,tb.u0.layer2.m_data,tb.u0.layer2.OCMUX);
    for (i=0; i<tb.u0.layer2.OCHAN/tb.u0.layer2.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer2.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation2[tb.u0.layer2.m_row*tb.u0.layer2.OWIDTH+tb.u0.layer2.m_col][((tb.u0.layer2.OCHAN-1)-i*tb.u0.layer2.OCMUX-tb.u0.layer2.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer2.m_data[i*DTYPE +:DTYPE],activation2[tb.u0.layer2.m_row*tb.u0.layer2.OWIDTH+tb.u0.layer2.m_col][((tb.u0.layer2.OCHAN-1)-i*tb.u0.layer2.OCMUX-tb.u0.layer2.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer2.m_data[i*DTYPE +:DTYPE],activation2[tb.u0.layer2.m_row*tb.u0.layer2.OWIDTH+tb.u0.layer2.m_col][((tb.u0.layer2.OCHAN-1)-i*tb.u0.layer2.OCMUX-tb.u0.layer2.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer3.m_valid) begin
    $display("LAYER3 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer3.m_row,tb.u0.layer3.m_col,tb.u0.layer3.m_chan,tb.u0.layer3.m_data,tb.u0.layer3.OCMUX);
    for (i=0; i<tb.u0.layer3.OCHAN/tb.u0.layer3.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer3.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation3[tb.u0.layer3.m_row*tb.u0.layer3.OWIDTH+tb.u0.layer3.m_col][((tb.u0.layer3.OCHAN-1)-i*tb.u0.layer3.OCMUX-tb.u0.layer3.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer3.m_data[i*DTYPE +:DTYPE],activation3[tb.u0.layer3.m_row*tb.u0.layer3.OWIDTH+tb.u0.layer3.m_col][((tb.u0.layer3.OCHAN-1)-i*tb.u0.layer3.OCMUX-tb.u0.layer3.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer3.m_data[i*DTYPE +:DTYPE],activation3[tb.u0.layer3.m_row*tb.u0.layer3.OWIDTH+tb.u0.layer3.m_col][((tb.u0.layer3.OCHAN-1)-i*tb.u0.layer3.OCMUX-tb.u0.layer3.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer4.m_valid) begin
    $display("LAYER4 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer4.m_row,tb.u0.layer4.m_col,tb.u0.layer4.m_chan,tb.u0.layer4.m_data,tb.u0.layer4.OCMUX);
    for (i=0; i<tb.u0.layer4.OCHAN/tb.u0.layer4.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer4.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation4[tb.u0.layer4.m_row*tb.u0.layer4.OWIDTH+tb.u0.layer4.m_col][((tb.u0.layer4.OCHAN-1)-i*tb.u0.layer4.OCMUX-tb.u0.layer4.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer4.m_data[i*DTYPE +:DTYPE],activation4[tb.u0.layer4.m_row*tb.u0.layer4.OWIDTH+tb.u0.layer4.m_col][((tb.u0.layer4.OCHAN-1)-i*tb.u0.layer4.OCMUX-tb.u0.layer4.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer4.m_data[i*DTYPE +:DTYPE],activation4[tb.u0.layer4.m_row*tb.u0.layer4.OWIDTH+tb.u0.layer4.m_col][((tb.u0.layer4.OCHAN-1)-i*tb.u0.layer4.OCMUX-tb.u0.layer4.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer5.m_valid) begin
    $display("LAYER5 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer5.m_row,tb.u0.layer5.m_col,tb.u0.layer5.m_chan,tb.u0.layer5.m_data,tb.u0.layer5.OCMUX);
    for (i=0; i<tb.u0.layer5.OCHAN/tb.u0.layer5.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer5.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation5[tb.u0.layer5.m_row*tb.u0.layer5.OWIDTH+tb.u0.layer5.m_col][((tb.u0.layer5.OCHAN-1)-i*tb.u0.layer5.OCMUX-tb.u0.layer5.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer5.m_data[i*DTYPE +:DTYPE],activation5[tb.u0.layer5.m_row*tb.u0.layer5.OWIDTH+tb.u0.layer5.m_col][((tb.u0.layer5.OCHAN-1)-i*tb.u0.layer5.OCMUX-tb.u0.layer5.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer5.m_data[i*DTYPE +:DTYPE],activation5[tb.u0.layer5.m_row*tb.u0.layer5.OWIDTH+tb.u0.layer5.m_col][((tb.u0.layer5.OCHAN-1)-i*tb.u0.layer5.OCMUX-tb.u0.layer5.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer6.m_valid) begin
    $display("LAYER6 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer6.m_row,tb.u0.layer6.m_col,tb.u0.layer6.m_chan,tb.u0.layer6.m_data,tb.u0.layer6.OCMUX);
    for (i=0; i<tb.u0.layer6.OCHAN/tb.u0.layer6.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer6.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation6[tb.u0.layer6.m_row*tb.u0.layer6.OWIDTH+tb.u0.layer6.m_col][((tb.u0.layer6.OCHAN-1)-i*tb.u0.layer6.OCMUX-tb.u0.layer6.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer6.m_data[i*DTYPE +:DTYPE],activation6[tb.u0.layer6.m_row*tb.u0.layer6.OWIDTH+tb.u0.layer6.m_col][((tb.u0.layer6.OCHAN-1)-i*tb.u0.layer6.OCMUX-tb.u0.layer6.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer6.m_data[i*DTYPE +:DTYPE],activation6[tb.u0.layer6.m_row*tb.u0.layer6.OWIDTH+tb.u0.layer6.m_col][((tb.u0.layer6.OCHAN-1)-i*tb.u0.layer6.OCMUX-tb.u0.layer6.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer7.m_valid) begin
    $display("LAYER7 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer7.m_row,tb.u0.layer7.m_col,tb.u0.layer7.m_chan,tb.u0.layer7.m_data,tb.u0.layer7.OCMUX);
    for (i=0; i<tb.u0.layer7.OCHAN/tb.u0.layer7.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer7.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation7[tb.u0.layer7.m_row*tb.u0.layer7.OWIDTH+tb.u0.layer7.m_col][((tb.u0.layer7.OCHAN-1)-i*tb.u0.layer7.OCMUX-tb.u0.layer7.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer7.m_data[i*DTYPE +:DTYPE],activation7[tb.u0.layer7.m_row*tb.u0.layer7.OWIDTH+tb.u0.layer7.m_col][((tb.u0.layer7.OCHAN-1)-i*tb.u0.layer7.OCMUX-tb.u0.layer7.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer7.m_data[i*DTYPE +:DTYPE],activation7[tb.u0.layer7.m_row*tb.u0.layer7.OWIDTH+tb.u0.layer7.m_col][((tb.u0.layer7.OCHAN-1)-i*tb.u0.layer7.OCMUX-tb.u0.layer7.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer8.m_valid) begin
    $display("LAYER8 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer8.m_row,tb.u0.layer8.m_col,tb.u0.layer8.m_chan,tb.u0.layer8.m_data,tb.u0.layer8.OCMUX);
    for (i=0; i<tb.u0.layer8.OCHAN/tb.u0.layer8.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer8.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation8[tb.u0.layer8.m_row*tb.u0.layer8.OWIDTH+tb.u0.layer8.m_col][((tb.u0.layer8.OCHAN-1)-i*tb.u0.layer8.OCMUX-tb.u0.layer8.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer8.m_data[i*DTYPE +:DTYPE],activation8[tb.u0.layer8.m_row*tb.u0.layer8.OWIDTH+tb.u0.layer8.m_col][((tb.u0.layer8.OCHAN-1)-i*tb.u0.layer8.OCMUX-tb.u0.layer8.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer8.m_data[i*DTYPE +:DTYPE],activation8[tb.u0.layer8.m_row*tb.u0.layer8.OWIDTH+tb.u0.layer8.m_col][((tb.u0.layer8.OCHAN-1)-i*tb.u0.layer8.OCMUX-tb.u0.layer8.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer9.m_valid) begin
    $display("LAYER9 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer9.m_row,tb.u0.layer9.m_col,tb.u0.layer9.m_chan,tb.u0.layer9.m_data,tb.u0.layer9.OCMUX);
    for (i=0; i<tb.u0.layer9.OCHAN/tb.u0.layer9.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer9.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation9[tb.u0.layer9.m_row*tb.u0.layer9.OWIDTH+tb.u0.layer9.m_col][((tb.u0.layer9.OCHAN-1)-i*tb.u0.layer9.OCMUX-tb.u0.layer9.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer9.m_data[i*DTYPE +:DTYPE],activation9[tb.u0.layer9.m_row*tb.u0.layer9.OWIDTH+tb.u0.layer9.m_col][((tb.u0.layer9.OCHAN-1)-i*tb.u0.layer9.OCMUX-tb.u0.layer9.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer9.m_data[i*DTYPE +:DTYPE],activation9[tb.u0.layer9.m_row*tb.u0.layer9.OWIDTH+tb.u0.layer9.m_col][((tb.u0.layer9.OCHAN-1)-i*tb.u0.layer9.OCMUX-tb.u0.layer9.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer10.m_valid) begin
    $display("LAYER10 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer10.m_row,tb.u0.layer10.m_col,tb.u0.layer10.m_chan,tb.u0.layer10.m_data,tb.u0.layer10.OCMUX);
    for (i=0; i<tb.u0.layer10.OCHAN/tb.u0.layer10.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer10.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation10[tb.u0.layer10.m_row*tb.u0.layer10.OWIDTH+tb.u0.layer10.m_col][((tb.u0.layer10.OCHAN-1)-i*tb.u0.layer10.OCMUX-tb.u0.layer10.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer10.m_data[i*DTYPE +:DTYPE],activation10[tb.u0.layer10.m_row*tb.u0.layer10.OWIDTH+tb.u0.layer10.m_col][((tb.u0.layer10.OCHAN-1)-i*tb.u0.layer10.OCMUX-tb.u0.layer10.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer10.m_data[i*DTYPE +:DTYPE],activation10[tb.u0.layer10.m_row*tb.u0.layer10.OWIDTH+tb.u0.layer10.m_col][((tb.u0.layer10.OCHAN-1)-i*tb.u0.layer10.OCMUX-tb.u0.layer10.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer11.m_valid) begin
    $display("LAYER11 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer11.m_row,tb.u0.layer11.m_col,tb.u0.layer11.m_chan,tb.u0.layer11.m_data,tb.u0.layer11.OCMUX);
    for (i=0; i<tb.u0.layer11.OCHAN/tb.u0.layer11.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer11.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation11[tb.u0.layer11.m_row*tb.u0.layer11.OWIDTH+tb.u0.layer11.m_col][((tb.u0.layer11.OCHAN-1)-i*tb.u0.layer11.OCMUX-tb.u0.layer11.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer11.m_data[i*DTYPE +:DTYPE],activation11[tb.u0.layer11.m_row*tb.u0.layer11.OWIDTH+tb.u0.layer11.m_col][((tb.u0.layer11.OCHAN-1)-i*tb.u0.layer11.OCMUX-tb.u0.layer11.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer11.m_data[i*DTYPE +:DTYPE],activation11[tb.u0.layer11.m_row*tb.u0.layer11.OWIDTH+tb.u0.layer11.m_col][((tb.u0.layer11.OCHAN-1)-i*tb.u0.layer11.OCMUX-tb.u0.layer11.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer12.m_valid) begin
    $display("LAYER12 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer12.m_row,tb.u0.layer12.m_col,tb.u0.layer12.m_chan,tb.u0.layer12.m_data,tb.u0.layer12.OCMUX);
    for (i=0; i<tb.u0.layer12.OCHAN/tb.u0.layer12.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer12.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation12[tb.u0.layer12.m_row*tb.u0.layer12.OWIDTH+tb.u0.layer12.m_col][((tb.u0.layer12.OCHAN-1)-i*tb.u0.layer12.OCMUX-tb.u0.layer12.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer12.m_data[i*DTYPE +:DTYPE],activation12[tb.u0.layer12.m_row*tb.u0.layer12.OWIDTH+tb.u0.layer12.m_col][((tb.u0.layer12.OCHAN-1)-i*tb.u0.layer12.OCMUX-tb.u0.layer12.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer12.m_data[i*DTYPE +:DTYPE],activation12[tb.u0.layer12.m_row*tb.u0.layer12.OWIDTH+tb.u0.layer12.m_col][((tb.u0.layer12.OCHAN-1)-i*tb.u0.layer12.OCMUX-tb.u0.layer12.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer13.m_valid) begin
    $display("LAYER13 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer13.m_row,tb.u0.layer13.m_col,tb.u0.layer13.m_chan,tb.u0.layer13.m_data,tb.u0.layer13.OCMUX);
    for (i=0; i<tb.u0.layer13.OCHAN/tb.u0.layer13.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer13.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation13[tb.u0.layer13.m_row*tb.u0.layer13.OWIDTH+tb.u0.layer13.m_col][((tb.u0.layer13.OCHAN-1)-i*tb.u0.layer13.OCMUX-tb.u0.layer13.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer13.m_data[i*DTYPE +:DTYPE],activation13[tb.u0.layer13.m_row*tb.u0.layer13.OWIDTH+tb.u0.layer13.m_col][((tb.u0.layer13.OCHAN-1)-i*tb.u0.layer13.OCMUX-tb.u0.layer13.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer13.m_data[i*DTYPE +:DTYPE],activation13[tb.u0.layer13.m_row*tb.u0.layer13.OWIDTH+tb.u0.layer13.m_col][((tb.u0.layer13.OCHAN-1)-i*tb.u0.layer13.OCMUX-tb.u0.layer13.m_chan)*DTYPE +:DTYPE]);
    end
end

if (tb.u0.layer14.m_valid) begin
    $display("LAYER14 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer14.m_row,tb.u0.layer14.m_col,tb.u0.layer14.m_chan,tb.u0.layer14.m_data,tb.u0.layer14.OCMUX);
    for (i=0; i<tb.u0.layer14.OCHAN/tb.u0.layer14.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer14.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation14[tb.u0.layer14.m_row*tb.u0.layer14.OWIDTH+tb.u0.layer14.m_col][((tb.u0.layer14.OCHAN-1)-i*tb.u0.layer14.OCMUX-tb.u0.layer14.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer14.m_data[i*DTYPE +:DTYPE],activation14[tb.u0.layer14.m_row*tb.u0.layer14.OWIDTH+tb.u0.layer14.m_col][((tb.u0.layer14.OCHAN-1)-i*tb.u0.layer14.OCMUX-tb.u0.layer14.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer14.m_data[i*DTYPE +:DTYPE],activation14[tb.u0.layer14.m_row*tb.u0.layer14.OWIDTH+tb.u0.layer14.m_col][((tb.u0.layer14.OCHAN-1)-i*tb.u0.layer14.OCMUX-tb.u0.layer14.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer15.m_valid) begin
    $display("LAYER15 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer15.m_row,tb.u0.layer15.m_col,tb.u0.layer15.m_chan,tb.u0.layer15.m_data,tb.u0.layer15.OCMUX);
    for (i=0; i<tb.u0.layer15.OCHAN/tb.u0.layer15.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer15.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation15[tb.u0.layer15.m_row*tb.u0.layer15.OWIDTH+tb.u0.layer15.m_col][((tb.u0.layer15.OCHAN-1)-i*tb.u0.layer15.OCMUX-tb.u0.layer15.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer15.m_data[i*DTYPE +:DTYPE],activation15[tb.u0.layer15.m_row*tb.u0.layer15.OWIDTH+tb.u0.layer15.m_col][((tb.u0.layer15.OCHAN-1)-i*tb.u0.layer15.OCMUX-tb.u0.layer15.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer15.m_data[i*DTYPE +:DTYPE],activation15[tb.u0.layer15.m_row*tb.u0.layer15.OWIDTH+tb.u0.layer15.m_col][((tb.u0.layer15.OCHAN-1)-i*tb.u0.layer15.OCMUX-tb.u0.layer15.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer16.m_valid) begin
    $display("LAYER16 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer16.m_row,tb.u0.layer16.m_col,tb.u0.layer16.m_chan,tb.u0.layer16.m_data,tb.u0.layer16.OCMUX);
    for (i=0; i<tb.u0.layer16.OCHAN/tb.u0.layer16.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer16.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation16[tb.u0.layer16.m_row*tb.u0.layer16.OWIDTH+tb.u0.layer16.m_col][((tb.u0.layer16.OCHAN-1)-i*tb.u0.layer16.OCMUX-tb.u0.layer16.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer16.m_data[i*DTYPE +:DTYPE],activation16[tb.u0.layer16.m_row*tb.u0.layer16.OWIDTH+tb.u0.layer16.m_col][((tb.u0.layer16.OCHAN-1)-i*tb.u0.layer16.OCMUX-tb.u0.layer16.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer16.m_data[i*DTYPE +:DTYPE],activation16[tb.u0.layer16.m_row*tb.u0.layer16.OWIDTH+tb.u0.layer16.m_col][((tb.u0.layer16.OCHAN-1)-i*tb.u0.layer16.OCMUX-tb.u0.layer16.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer17.m_valid) begin
    $display("LAYER17 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer17.m_row,tb.u0.layer17.m_col,tb.u0.layer17.m_chan,tb.u0.layer17.m_data,tb.u0.layer17.OCMUX);
    for (i=0; i<tb.u0.layer17.OCHAN/tb.u0.layer17.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer17.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation17[tb.u0.layer17.m_row*tb.u0.layer17.OWIDTH+tb.u0.layer17.m_col][((tb.u0.layer17.OCHAN-1)-i*tb.u0.layer17.OCMUX-tb.u0.layer17.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer17.m_data[i*DTYPE +:DTYPE],activation17[tb.u0.layer17.m_row*tb.u0.layer17.OWIDTH+tb.u0.layer17.m_col][((tb.u0.layer17.OCHAN-1)-i*tb.u0.layer17.OCMUX-tb.u0.layer17.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer17.m_data[i*DTYPE +:DTYPE],activation17[tb.u0.layer17.m_row*tb.u0.layer17.OWIDTH+tb.u0.layer17.m_col][((tb.u0.layer17.OCHAN-1)-i*tb.u0.layer17.OCMUX-tb.u0.layer17.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer18.m_valid) begin
    $display("LAYER18 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer18.m_row,tb.u0.layer18.m_col,tb.u0.layer18.m_chan,tb.u0.layer18.m_data,tb.u0.layer18.OCMUX);
    for (i=0; i<tb.u0.layer18.OCHAN/tb.u0.layer18.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer18.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation18[tb.u0.layer18.m_row*tb.u0.layer18.OWIDTH+tb.u0.layer18.m_col][((tb.u0.layer18.OCHAN-1)-i*tb.u0.layer18.OCMUX-tb.u0.layer18.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer18.m_data[i*DTYPE +:DTYPE],activation18[tb.u0.layer18.m_row*tb.u0.layer18.OWIDTH+tb.u0.layer18.m_col][((tb.u0.layer18.OCHAN-1)-i*tb.u0.layer18.OCMUX-tb.u0.layer18.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer18.m_data[i*DTYPE +:DTYPE],activation18[tb.u0.layer18.m_row*tb.u0.layer18.OWIDTH+tb.u0.layer18.m_col][((tb.u0.layer18.OCHAN-1)-i*tb.u0.layer18.OCMUX-tb.u0.layer18.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer19.m_valid) begin
    $display("LAYER19 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer19.m_row,tb.u0.layer19.m_col,tb.u0.layer19.m_chan,tb.u0.layer19.m_data,tb.u0.layer19.OCMUX);
    for (i=0; i<tb.u0.layer19.OCHAN/tb.u0.layer19.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer19.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation19[tb.u0.layer19.m_row*tb.u0.layer19.OWIDTH+tb.u0.layer19.m_col][((tb.u0.layer19.OCHAN-1)-i*tb.u0.layer19.OCMUX-tb.u0.layer19.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer19.m_data[i*DTYPE +:DTYPE],activation19[tb.u0.layer19.m_row*tb.u0.layer19.OWIDTH+tb.u0.layer19.m_col][((tb.u0.layer19.OCHAN-1)-i*tb.u0.layer19.OCMUX-tb.u0.layer19.m_chan)*DTYPE +:DTYPE]);
        end
//        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer19.m_data[i*DTYPE +:DTYPE],activation19[tb.u0.layer19.m_row*tb.u0.layer19.OWIDTH+tb.u0.layer19.m_col][((tb.u0.layer19.OCHAN-1)-i*tb.u0.layer19.OCMUX-tb.u0.layer19.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer20.m_valid) begin
    $display("LAYER20 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer20.m_row,tb.u0.layer20.m_col,tb.u0.layer20.m_chan,tb.u0.layer20.m_data,tb.u0.layer20.OCMUX);
    for (i=0; i<tb.u0.layer20.OCHAN/tb.u0.layer20.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer20.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation20[tb.u0.layer20.m_row*tb.u0.layer20.OWIDTH+tb.u0.layer20.m_col][((tb.u0.layer20.OCHAN-1)-i*tb.u0.layer20.OCMUX-tb.u0.layer20.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer20.m_data[i*DTYPE +:DTYPE],activation20[tb.u0.layer20.m_row*tb.u0.layer20.OWIDTH+tb.u0.layer20.m_col][((tb.u0.layer20.OCHAN-1)-i*tb.u0.layer20.OCMUX-tb.u0.layer20.m_chan)*DTYPE +:DTYPE]);
        end
        //else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer20.m_data[i*DTYPE +:DTYPE],activation20[tb.u0.layer20.m_row*tb.u0.layer20.OWIDTH+tb.u0.layer20.m_col][((tb.u0.layer20.OCHAN-1)-i*tb.u0.layer20.OCMUX-tb.u0.layer20.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer21.m_valid) begin
    $display("LAYER21 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer21.m_row,tb.u0.layer21.m_col,tb.u0.layer21.m_chan,tb.u0.layer21.m_data,tb.u0.layer21.OCMUX);
    for (i=0; i<tb.u0.layer21.OCHAN/tb.u0.layer21.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer21.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation21[tb.u0.layer21.m_row*tb.u0.layer21.OWIDTH+tb.u0.layer21.m_col][((tb.u0.layer21.OCHAN-1)-i*tb.u0.layer21.OCMUX-tb.u0.layer21.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer21.m_data[i*DTYPE +:DTYPE],activation21[tb.u0.layer21.m_row*tb.u0.layer21.OWIDTH+tb.u0.layer21.m_col][((tb.u0.layer21.OCHAN-1)-i*tb.u0.layer21.OCMUX-tb.u0.layer21.m_chan)*DTYPE +:DTYPE]);
        end
        //else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer21.m_data[i*DTYPE +:DTYPE],activation21[tb.u0.layer21.m_row*tb.u0.layer21.OWIDTH+tb.u0.layer21.m_col][((tb.u0.layer21.OCHAN-1)-i*tb.u0.layer21.OCMUX-tb.u0.layer21.m_chan)*DTYPE +:DTYPE]);
    end
end
if (tb.u0.layer22.m_valid) begin
    $display("LAYER22 time %f m_row %d m_col %d m_chan %d m_data %h ocmux %d",$realtime,tb.u0.layer22.m_row,tb.u0.layer22.m_col,tb.u0.layer22.m_chan,tb.u0.layer22.m_data,tb.u0.layer22.OCMUX);
    for (i=0; i<tb.u0.layer22.OCHAN/tb.u0.layer22.OCMUX; i=i+1) begin
        if ($pow($bitstoshortreal(tb.u0.layer22.m_data[i*DTYPE +:DTYPE])-$bitstoshortreal(activation22[tb.u0.layer22.m_row*tb.u0.layer22.OWIDTH+tb.u0.layer22.m_col][((tb.u0.layer22.OCHAN-1)-i*tb.u0.layer22.OCMUX-tb.u0.layer22.m_chan)*DTYPE +:DTYPE]),2) > TOLERANCE) begin
            $display("FAIL i %d m_data %h activation %h",i,tb.u0.layer22.m_data[i*DTYPE +:DTYPE],activation22[tb.u0.layer22.m_row*tb.u0.layer22.OWIDTH+tb.u0.layer22.m_col][((tb.u0.layer22.OCHAN-1)-i*tb.u0.layer22.OCMUX-tb.u0.layer22.m_chan)*DTYPE +:DTYPE]);
        end
        else $display("PASS i %d m_data %h activation %h",i,tb.u0.layer22.m_data[i*DTYPE +:DTYPE],activation22[tb.u0.layer22.m_row*tb.u0.layer22.OWIDTH+tb.u0.layer22.m_col][((tb.u0.layer22.OCHAN-1)-i*tb.u0.layer22.OCMUX-tb.u0.layer22.m_chan)*DTYPE +:DTYPE]);
    end
end

end
endmodule
