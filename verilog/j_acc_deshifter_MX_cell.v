module j_acc_deshifter_MX_cell #(
  parameter SRAM_DEPTH  = 256*256*4        ,
  parameter SRAM_ADDR_W = clog2(SRAM_DEPTH)
) (
  input                            clk          ,
  input                            reset_n      ,
  output wire                      sram_en      ,
  output wire [   SRAM_ADDR_W-1:0] sram_addr    ,
  output wire [              31:0] sram_data    ,
  input                            shift_start  ,
  output wire                      shift_start_o,
  output                           shift_idle   ,
  input  wire [SRAM_ADDR_W*32-1:0] start_addr   ,
  input  wire [   SRAM_ADDR_W-1:0] img_size     ,
  input  wire [          32*1-1:0] serial_input ,
  input  wire [          32*1-1:0] serial_en
);
function [31:0] clog2 (input [31:0] x);
    reg [31:0] x_tmp;
    begin
            x_tmp = x-1;
            for(clog2=0; x_tmp>0; clog2=clog2+1) begin
                x_tmp = x_tmp >> 1;
            end
    end
endfunction

wire [          1*32- 1:0] mx_sram_en    ;
wire [SRAM_ADDR_W*32- 1:0] mx_sram_addr  ;
reg  [          1*32- 1:0] mx_shift_start;
wire [          1*32- 1:0] mx_shift_idle ;
wire [          32*32-1:0] mx_sram_data  ;

always @(posedge clk) begin : proc_mx_shift_start
  if(~reset_n) begin
    mx_shift_start <= 32'b0;
  end else begin
    mx_shift_start <= {mx_shift_start[30:0], shift_start};
  end
end

assign shift_start_o = mx_shift_start[31];

genvar i;
generate
  for (i = 0; i < 32; i=i+1) begin: g_acc_deshifter
    j_acc_deshifter #(.SRAM_DEPTH(SRAM_DEPTH)) i_j_acc_deshifter (
      .clk         (clk                                            ),
      .reset_n     (reset_n                                        ),
      .sram_en     (mx_sram_en       [i]                           ),
      .sram_addr   (mx_sram_addr     [i*SRAM_ADDR_W +: SRAM_ADDR_W]),
      .sram_data   (mx_sram_data     [i*32          +: 32         ]),
      .shift_start (mx_shift_start   [i]                           ),
      .shift_idle  (mx_shift_idle    [i]                           ),
      .start_addr  (start_addr       [i*SRAM_ADDR_W +: SRAM_ADDR_W]),
      .img_size    (img_size                                       ),
      .serial_input(serial_input     [i]                           ),
      .serial_en   (serial_en        [i]                           )
    );
  end
endgenerate

assign sram_en   = (|mx_sram_en);
assign sram_addr = {(SRAM_ADDR_W){mx_sram_en[00]}} & mx_sram_addr[00*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[01]}} & mx_sram_addr[01*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[02]}} & mx_sram_addr[02*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[03]}} & mx_sram_addr[03*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[04]}} & mx_sram_addr[04*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[05]}} & mx_sram_addr[05*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[06]}} & mx_sram_addr[06*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[07]}} & mx_sram_addr[07*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[08]}} & mx_sram_addr[08*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[09]}} & mx_sram_addr[09*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[10]}} & mx_sram_addr[10*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[11]}} & mx_sram_addr[11*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[12]}} & mx_sram_addr[12*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[13]}} & mx_sram_addr[13*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[14]}} & mx_sram_addr[14*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[15]}} & mx_sram_addr[15*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[16]}} & mx_sram_addr[16*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[17]}} & mx_sram_addr[17*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[18]}} & mx_sram_addr[18*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[19]}} & mx_sram_addr[19*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[20]}} & mx_sram_addr[20*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[21]}} & mx_sram_addr[21*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[22]}} & mx_sram_addr[22*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[23]}} & mx_sram_addr[23*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[24]}} & mx_sram_addr[24*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[25]}} & mx_sram_addr[25*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[26]}} & mx_sram_addr[26*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[27]}} & mx_sram_addr[27*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[28]}} & mx_sram_addr[28*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[29]}} & mx_sram_addr[29*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[30]}} & mx_sram_addr[30*SRAM_ADDR_W +: SRAM_ADDR_W]|
                   {(SRAM_ADDR_W){mx_sram_en[31]}} & mx_sram_addr[31*SRAM_ADDR_W +: SRAM_ADDR_W];


assign sram_data = {(32){mx_sram_en[00]}} & mx_sram_data[00*32 +: 32]|
                   {(32){mx_sram_en[01]}} & mx_sram_data[01*32 +: 32]|
                   {(32){mx_sram_en[02]}} & mx_sram_data[02*32 +: 32]|
                   {(32){mx_sram_en[03]}} & mx_sram_data[03*32 +: 32]|
                   {(32){mx_sram_en[04]}} & mx_sram_data[04*32 +: 32]|
                   {(32){mx_sram_en[05]}} & mx_sram_data[05*32 +: 32]|
                   {(32){mx_sram_en[06]}} & mx_sram_data[06*32 +: 32]|
                   {(32){mx_sram_en[07]}} & mx_sram_data[07*32 +: 32]|
                   {(32){mx_sram_en[08]}} & mx_sram_data[08*32 +: 32]|
                   {(32){mx_sram_en[09]}} & mx_sram_data[09*32 +: 32]|
                   {(32){mx_sram_en[10]}} & mx_sram_data[10*32 +: 32]|
                   {(32){mx_sram_en[11]}} & mx_sram_data[11*32 +: 32]|
                   {(32){mx_sram_en[12]}} & mx_sram_data[12*32 +: 32]|
                   {(32){mx_sram_en[13]}} & mx_sram_data[13*32 +: 32]|
                   {(32){mx_sram_en[14]}} & mx_sram_data[14*32 +: 32]|
                   {(32){mx_sram_en[15]}} & mx_sram_data[15*32 +: 32]|
                   {(32){mx_sram_en[16]}} & mx_sram_data[16*32 +: 32]|
                   {(32){mx_sram_en[17]}} & mx_sram_data[17*32 +: 32]|
                   {(32){mx_sram_en[18]}} & mx_sram_data[18*32 +: 32]|
                   {(32){mx_sram_en[19]}} & mx_sram_data[19*32 +: 32]|
                   {(32){mx_sram_en[20]}} & mx_sram_data[20*32 +: 32]|
                   {(32){mx_sram_en[21]}} & mx_sram_data[21*32 +: 32]|
                   {(32){mx_sram_en[22]}} & mx_sram_data[22*32 +: 32]|
                   {(32){mx_sram_en[23]}} & mx_sram_data[23*32 +: 32]|
                   {(32){mx_sram_en[24]}} & mx_sram_data[24*32 +: 32]|
                   {(32){mx_sram_en[25]}} & mx_sram_data[25*32 +: 32]|
                   {(32){mx_sram_en[26]}} & mx_sram_data[26*32 +: 32]|
                   {(32){mx_sram_en[27]}} & mx_sram_data[27*32 +: 32]|
                   {(32){mx_sram_en[28]}} & mx_sram_data[28*32 +: 32]|
                   {(32){mx_sram_en[29]}} & mx_sram_data[29*32 +: 32]|
                   {(32){mx_sram_en[30]}} & mx_sram_data[30*32 +: 32]|
                   {(32){mx_sram_en[31]}} & mx_sram_data[31*32 +: 32];


assign shift_idle = &mx_shift_idle;


endmodule      
