module j_wgt_shifter #(
  parameter SRAM_DEPTH  = 256*256*4        ,
  parameter SHIFT_WIDTH = 8                ,
  parameter SRAM_ADDR_W = clog2(SRAM_DEPTH)
) (
  input                         clk          ,
  input                         reset_n      ,
  output reg                    sram_en      ,
  output reg  [SRAM_ADDR_W-1:0] sram_addr    ,
  input       [            7:0] sram_data    ,
  input                         shift_start  ,
  output                        shift_idle   ,
  input  wire [SRAM_ADDR_W-1:0] end_addr     ,
  input  wire [SRAM_ADDR_W-1:0] img_size     ,
  output wire [SHIFT_WIDTH-1:0] serial_output,
  output reg                    serial_start ,
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

	reg  [            2:0] fsm_state       ;
	reg  [            2:0] fsm_state_nxt   ;
	reg  [            2:0] shift_cnt       ;
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
        end else if(SHIFT_WIDTH == 8) begin
          fsm_state_nxt = S_SHIFT;
        end else if(shift_cnt_done) begin
          fsm_state_nxt = S_LOAD_DATA;
        end
      end
      default : /* default */;
    endcase
  end

  assign cur_cnt_inc = (img_cnt_inc & (SHIFT_WIDTH<=2)) | (sram_en & (SHIFT_WIDTH>=4));
  assign cur_cnt_clr = (cur_cnt == img_size) & img_cnt_inc & ((SHIFT_WIDTH<=2) | 
                                                              ((SHIFT_WIDTH>=4)&(fsm_state==S_SHIFT))
                                                             );
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
  assign shift_cnt_done = SHIFT_WIDTH == 1 ? shift_cnt == 3'b101 :
                          SHIFT_WIDTH == 2 ? shift_cnt == 3'b010 :
                          SHIFT_WIDTH == 4 ? 1                   :
                          SHIFT_WIDTH == 8 ? 1                   : 0;

  assign shift_cnt_inc  = ~(shift_cnt == 3'b0) | shift_state_dly;
  assign img_cnt_inc    = shift_cnt_done;
  always @(posedge clk) begin : proc_shift_state_dly
    if(~reset_n) begin
      shift_state_dly <= 0;
    end else begin
      shift_state_dly <= (fsm_state == S_SHIFT);
    end
  end

  wire serial_start_nxt;
  assign serial_start_nxt = shift_cnt_inc & (shift_cnt == 3'b0);

  always @(posedge clk) begin : proc_serial_start
  	if(~reset_n) begin
  		serial_start <= 0;
  	end else begin
  		serial_start <= serial_start_nxt;
  	end
  end

  wire [2:0] shift_cnt_nxt = shift_cnt + (SHIFT_WIDTH == 1 ? 1 :
                                          SHIFT_WIDTH == 2 ? 2 :
                                          SHIFT_WIDTH == 4 ? 4 :
                                          SHIFT_WIDTH == 8 ? 0 : 0
                                          );
  always @(posedge clk) begin : proc_shift_cnt
    if(~reset_n) begin
      shift_cnt <= 0;
    end else if(shift_cnt_inc) begin
      shift_cnt <= shift_cnt_nxt;
    end
  end

  assign fake_sram_en_nxt = (fsm_state == S_LOAD_DATA) | shift_start | 
                            (SHIFT_WIDTH == 8) & (fsm_state == S_SHIFT) & ~cur_cnt_clr;
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

  reg  [7:0] sram_data_latch;
  wire       zero_skip      ;

  assign zero_skip = 1'b0;

  always @(posedge clk) begin : proc_sram_data_latch
    if(~reset_n) begin
      sram_data_latch <= 0;
    end else if(fake_sram_en_dly) begin
      sram_data_latch <= sram_data[7:0];
    end else if(shift_cnt_inc_dly0) begin
      case(SHIFT_WIDTH)
        1: sram_data_latch <= {1'b0, sram_data_latch[7:1]};
        2: sram_data_latch <= {2'b0, sram_data_latch[7:2]};
        4: sram_data_latch <= {4'b0, sram_data_latch[7:4]};
      endcase
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
          // latch first point
          sram_addr_nxt = end_addr;
          sram_en_nxt   = 1'b1;
      end else if(fsm_state == S_LOAD_DATA) begin
          sram_addr_nxt = sram_addr - 1;
          sram_en_nxt   = 1'b1;
      end else if((SHIFT_WIDTH==8) & (fsm_state == S_SHIFT)) begin
          sram_addr_nxt = sram_addr - 1;
          sram_en_nxt   = (1'b1) & ~cur_cnt_clr;
      end
  end

  always @(posedge clk) begin: proc_sram_addr
    if(~reset_n) begin
      sram_addr <= 0;
    end begin
      sram_addr <= sram_addr_nxt;
    end
  end

  assign serial_output = sram_data_latch[SHIFT_WIDTH-1:0];
  assign serial_en     = shift_cnt_inc_dly0;
  assign shift_idle    = (fsm_state == S_IDEL) & (shift_cnt == 5'b00000);

endmodule      
