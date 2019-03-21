module j_acc_shifter #(
	parameter SRAM_DEPTH  = 256*256*4        ,
	parameter SRAM_ADDR_W = clog2(SRAM_DEPTH)
) (
	input                         clk          ,
	input                         reset_n      ,
	output reg                    sram_en      ,
	output reg  [SRAM_ADDR_W-1:0] sram_addr    ,
	input       [           31:0] sram_data    ,
	input                         shift_start  ,
	output                        shift_idle   ,
	input                         shift_ctrl   ,
	input  wire [SRAM_ADDR_W-1:0] start_addr   ,
	input  wire [SRAM_ADDR_W-1:0] img_size     ,
	output wire                   serial_output,
	output reg                    serial_start ,
	output reg                    serial_end   ,
	output wire                   serial_en
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

  parameter S_IDEL              = 0;
  parameter S_LOAD_DATA         = 1;
  parameter S_SHIFT             = 2;

  parameter SHT_KEEP       = 0;
  parameter SHT_ZERO       = 1;

	reg  [            2:0] fsm_state       ;
	reg  [            2:0] fsm_state_nxt   ;
	reg  [            4:0] shift_cnt       ;
	wire                   shift_cnt_inc   ;
	wire                   shift_cnt_done  ;
	reg  [SRAM_ADDR_W-1:0] cur_cnt         ;
	wire [SRAM_ADDR_W-1:0] cur_cnt_nxt     ;
	wire                   cur_cnt_inc     ;
	wire                   cur_cnt_clr     ;
	reg                    fake_sram_en, fake_sram_en_dly;
	wire                   fake_sram_en_nxt;
	wire                   img_cnt_inc     ;

  always @(posedge clk) begin : proc_fsm_state
    if(~reset_n) begin
      fsm_state <= S_IDEL;
    end else begin
      fsm_state <= fsm_state_nxt;
    end
  end

  always @(*) begin : proc_fsm_state_nxt
    fsm_state_nxt = fsm_state;
    case (fsm_state)
      S_IDEL: begin
        if(shift_start)
          fsm_state_nxt = S_SHIFT;
      end
      S_LOAD_DATA: begin
        fsm_state_nxt = S_SHIFT;
      end
      S_SHIFT: begin
        if(cur_cnt_clr) begin
          fsm_state_nxt = S_IDEL;
        end else if(shift_cnt_done) begin
          fsm_state_nxt = S_LOAD_DATA;
        end
      end
      default : /* default */;
    endcase
  end

  assign cur_cnt_inc = img_cnt_inc;
  assign cur_cnt_clr = (cur_cnt == img_size) & img_cnt_inc;
  assign cur_cnt_nxt = cur_cnt+1;

  always @(posedge clk) begin : proc_cur_cnt
    if(~reset_n) begin
      cur_cnt <= 0;
    end else if(cur_cnt_clr) begin
      cur_cnt <= 0;
    end else if(cur_cnt_inc) begin
      cur_cnt <= cur_cnt_nxt;
    end
  end

  // Shift control
  reg shift_state_dly;
  assign shift_cnt_done = shift_cnt == 5'b11101;
  assign shift_cnt_inc  = ~(shift_cnt == 5'b0) | shift_state_dly;
  assign img_cnt_inc    = shift_cnt == 5'b11101;
  always @(posedge clk) begin : proc_shift_state_dly
    if(~reset_n) begin
      shift_state_dly <= 0;
    end else begin
      shift_state_dly <= (fsm_state == S_SHIFT);
    end
  end

  wire serial_start_nxt;
  assign serial_start_nxt = shift_cnt_inc & (shift_cnt == 5'b0);

  always @(posedge clk) begin : proc_serial_start
  	if(~reset_n) begin
  		serial_start <= 0;
  	end else begin
  		serial_start <= serial_start_nxt;
  	end
  end

  wire serial_end_nxt;
  assign serial_end_nxt = (shift_cnt == 5'b11111);
  always @(posedge clk) begin : proc_serial_end
  	if(~reset_n) begin
  		serial_end <= 0;
  	end else begin
  		serial_end <= serial_end_nxt;
  	end
  end

  wire [4:0] shift_cnt_nxt = shift_cnt + 1;
  always @(posedge clk) begin : proc_shift_cnt
    if(~reset_n) begin
      shift_cnt <= 0;
    end else if(shift_cnt_inc) begin
      shift_cnt <= shift_cnt_nxt;
    end
  end

  assign fake_sram_en_nxt = (fsm_state == S_LOAD_DATA) | shift_start;
  always @(posedge clk) begin : proc_fake_sram_en
    if(~reset_n) begin
      {fake_sram_en_dly, fake_sram_en} <= 0;
    end else begin
      {fake_sram_en_dly, fake_sram_en} <= {fake_sram_en, fake_sram_en_nxt};
    end
  end

  reg sram_en_dly;
  wire sram_en_dly_nxt = sram_en;
  always @(posedge clk) begin : proc_sram_en_dly
    if(~reset_n) begin
      sram_en_dly <= 0;
    end else begin
      sram_en_dly <= sram_en_dly_nxt;
    end
  end

  reg shift_cnt_inc_dly0,shift_cnt_inc_dly1;
  always @(posedge clk) begin : proc_shift_cnt_inc_dly
    if(~reset_n) begin
      {shift_cnt_inc_dly1, shift_cnt_inc_dly0} <= 0;
    end else begin
      {shift_cnt_inc_dly1, shift_cnt_inc_dly0} <= {shift_cnt_inc_dly0, shift_cnt_inc};
    end
  end

  reg  [31:0] sram_data_latch;
  wire        zero_skip      ;

  assign zero_skip = 
        shift_ctrl == SHT_KEEP ? 1'b0:
        shift_ctrl == SHT_ZERO ? 1'b1:
        1'b0;

  always @(posedge clk) begin : proc_sram_data_latch
    if(~reset_n) begin
      sram_data_latch <= 0;
    end else if(fake_sram_en_dly) begin
      sram_data_latch <= zero_skip ? 32'b0: sram_data[31:0];
    end else if(shift_cnt_inc_dly0) begin
      sram_data_latch <= {1'b0, sram_data_latch[31:1]};
    end
  end

  reg sram_en_nxt;
  always @(posedge clk) begin : proc_sram_en
    if(~reset_n) begin
      sram_en <= 0;
    end else begin
      sram_en <= sram_en_nxt;
    end
  end

  reg [SRAM_ADDR_W-1:0] sram_addr_nxt;
 
  always @(*) begin : proc_sram_addr_nxt
      sram_addr_nxt     = sram_addr;
      sram_en_nxt       = 1'b0;
      if((shift_start == 1'b1) & (fsm_state == S_IDEL)) begin
        case(shift_ctrl)
          SHT_KEEP       : begin
            // latch first point
            sram_addr_nxt = start_addr;
            sram_en_nxt   = 1'b1;
          end
          SHT_ZERO       : begin
            // latch first point
            sram_addr_nxt = 0;
            sram_en_nxt   = 1'b0;
          end
        endcase // shift_ctrl
      end else if(fsm_state == S_LOAD_DATA) begin
        case(shift_ctrl)
          SHT_KEEP       : begin
            sram_addr_nxt = sram_addr + 1;
            sram_en_nxt   = 1'b1;
          end
          SHT_ZERO       : begin
            sram_addr_nxt = 0;
            sram_en_nxt   = 1'b0;
          end
        endcase // shift_ctrl      
      end
  end

  always @(posedge clk) begin: proc_sram_addr
    if(~reset_n) begin
      sram_addr <= 0;
    end begin
      sram_addr <= sram_addr_nxt;
    end
  end

  assign serial_output = sram_data_latch[0];
  assign serial_en     = shift_cnt_inc_dly0;
  assign shift_idle    = (fsm_state == S_IDEL) & (shift_cnt == 5'b00000);

endmodule      
