module tb_shifter_MX_cell;

	parameter IMG_W       = 32               ;
	parameter IMG_H       = 32               ;
	parameter SRAM_DEPTH  = 32*32            ;
	parameter SRAM_ADDR_W = clog2(SRAM_DEPTH);

function [31:0] clog2 (input [31:0] x);
    reg [31:0] x_tmp;
    begin
            x_tmp = x-1;
            for(clog2=0; x_tmp>0; clog2=clog2+1) begin
                x_tmp = x_tmp >> 1;
            end
    end
endfunction

	logic                         clk             = 'b0                    ;
	logic                         reset_n         = 'b0                    ;
	logic [                  8:0] sram_data       = 'b0                    ;
	logic                         shift_start     = 'b0                    ;
	logic [              4*8-1:0] shift_ctrl      = 'b0                    ;
	logic [    SRAM_ADDR_W*8-1:0] start_addr      = {(SRAM_ADDR_W*8){1'b0}};
	logic [      SRAM_ADDR_W-1:0] img_width_size  = 'b0                    ;
	logic [      SRAM_ADDR_W-1:0] img_height_size = 'b0                    ;
	wire                          sram_en                                  ;
	wire  [clog2(SRAM_DEPTH)-1:0] sram_addr                                ;
	wire                          shift_idle                               ;
	wire  [                  7:0] serial_output                            ;
	wire  [                  7:0] serial_en                                ;

	logic                         we    ;
	logic [clog2(SRAM_DEPTH)-1:0] addr  ;
	logic [                  8:0] data_i;
	wire  [                  8:0] data_o;

	sram #(.DATA_WIDTH(9), .ADDR_WIDTH(clog2(SRAM_DEPTH)), .RAM_SIZE(SRAM_DEPTH)) i_sram (
		.clk   (clk      ),
		.we    (1'b0     ),
		.en    (sram_en  ),
		.addr  (sram_addr),
		.data_i(9'b0     ),
		.data_o(sram_data)
	);

	j_shifter_MX_cell #(.SRAM_DEPTH(SRAM_DEPTH)) i_j_shifter_MX_cell (
		.clk            (clk            ),
		.reset_n        (reset_n        ),
		.sram_en        (sram_en        ),
		.sram_addr      (sram_addr      ),
		.sram_data      (sram_data      ),
		.shift_start    (shift_start    ),
		.shift_idle     (shift_idle     ),
		.shift_ctrl     (shift_ctrl     ),
		.start_addr     (start_addr     ),
		.img_width_size (img_width_size ),
		.img_height_size(img_height_size),
		.serial_output  (serial_output  ),
		.serial_en      (serial_en      )
	);

	initial begin
	  $dumpfile("test.vcd");
	  $dumpvars;
	end

	initial begin
		clk = 0;
		forever #5 clk = ~clk;
	end

  parameter SHT_KEEP       = 0;
  parameter SHT_UP         = 1;
  parameter SHT_DOWN       = 2;
  parameter SHT_LEFT       = 3;
  parameter SHT_RIGHT      = 4;
  parameter SHT_RIGHT_UP   = 5;
  parameter SHT_RIGHT_DOWN = 6;
  parameter SHT_LEFT_DOWN  = 7;
  parameter SHT_LEFT_UP    = 8;

	initial begin
		reset_n = 1'b1;
		@(posedge clk);
		reset_n = 1'b0;
		@(posedge clk);
		@(posedge clk);
		reset_n = 1'b1;
	end

	logic [7:0] result_queue[8][$];
	logic [7:0] check_queue [8][$];

	// serial part
	task automatic get_clk();
		@(posedge clk);
	endtask : get_clk

	task automatic get_result_bit(output logic result_o, input int idx);
		@(posedge clk);
		result_o = serial_output[idx];
	endtask : get_result_bit

	task automatic get_result(output logic [7:0] result_o, input int idx);
		for (int i = 0; i < 8; i = ((serial_en[idx]==1) ? i+1 : i)) begin
			get_result_bit(.result_o(result_o[i]), .idx(idx));
		end
	endtask : get_result

	task automatic check_8(input logic [7:0] golden, input logic [7:0] check_result, input string err_msg="");
		if(golden !== check_result) begin
			$error($stime, " Error golden 0x%h != got 0x%h", golden, check_result);
			$finish;
		end
	endtask : check_8




	task automatic generate_check(input logic [3:0] ctrl, input logic [31:0] start_addr, input int idx_c);
		logic [3:0]  shift_ctrl;
		logic        pre_zero_skip;
		logic [31:0] cur_addr;

		shift_ctrl = ctrl;
		pre_zero_skip = 0;
		case(shift_ctrl)
			SHT_KEEP: begin
				cur_addr = start_addr;
				for (int cur_h = 0; cur_h <= img_height_size; cur_h++) begin
					for (int cur_w = 0; cur_w <= img_width_size; cur_w++) begin
						if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue[idx_c].push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue[idx_c].push_back(i_sram.RAM[cur_addr][7:0]);
							pre_zero_skip = i_sram.RAM[cur_addr][8];
							cur_addr ++;
						end
					end
				end
			end

		SHT_UP: begin
				cur_addr = start_addr+img_width_size+1;
				for (int cur_h = 0; cur_h <= img_height_size; cur_h++) begin
					for (int cur_w = 0; cur_w <= img_width_size; cur_w++) begin
						if(cur_h == 0) begin
							$display($stime, " push addr[%h] = up zero", cur_addr);
							check_queue[idx_c].push_back(0);
						end else if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue[idx_c].push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue[idx_c].push_back(i_sram.RAM[cur_addr][7:0]);
							pre_zero_skip = i_sram.RAM[cur_addr][8];
							cur_addr ++;
						end
					end
				end
			end

		SHT_DOWN: begin
				cur_addr = start_addr+img_width_size+1;
				for (int cur_h = 0; cur_h <= img_height_size; cur_h++) begin
					for (int cur_w = 0; cur_w <= img_width_size; cur_w++) begin
						if(cur_h == img_height_size) begin
							$display($stime, " push addr[%h] = down zero", cur_addr);
							check_queue[idx_c].push_back(0);
						end else if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue[idx_c].push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue[idx_c].push_back(i_sram.RAM[cur_addr][7:0]);
							pre_zero_skip = i_sram.RAM[cur_addr][8];
							cur_addr ++;
						end
					end
				end
			end

		SHT_LEFT: begin
				cur_addr = start_addr;
				for (int cur_h = 0; cur_h <= img_height_size; cur_h++) begin
					for (int cur_w = 0; cur_w <= img_width_size; cur_w++) begin
						if((cur_w == 0) & (cur_h == 0)) begin
							$display($stime, " push addr[%h] = left zero", cur_addr);
							check_queue[idx_c].push_back(0);
						end else if(cur_w == 0) begin
							 $display($stime, " push addr[%h] = left zero", cur_addr);
							 check_queue[idx_c].push_back(0);
							 // if(pre_zero_skip == 1) begin
							 // 	cur_addr = cur_addr + 2;
							 // end else begin
							 // 	cur_addr = cur_addr + 1;
							 // end
							 // pre_zero_skip = 0;
						end else if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue[idx_c].push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue[idx_c].push_back(i_sram.RAM[cur_addr][7:0]);
							pre_zero_skip = i_sram.RAM[cur_addr][8];
							if(pre_zero_skip) $display($stime, " addr=%h have zero skip", cur_addr);
							cur_addr ++;
						end
					end
				end
			end

		SHT_RIGHT: begin
				cur_addr = start_addr+1;
				for (int cur_h = 0; cur_h <= img_height_size; cur_h++) begin
					for (int cur_w = 0; cur_w <= img_width_size; cur_w++) begin
						if(cur_w == img_width_size) begin
							 $display($stime, " push addr[%h] = right zero", cur_addr);
							 check_queue[idx_c].push_back(0);
							 cur_addr ++;
							 // if(pre_zero_skip == 1) begin
							 // 	cur_addr = cur_addr + 2;
							 // end else begin
							 // 	cur_addr = cur_addr + 1;
							 // end
							 // pre_zero_skip = 0;
						end else if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue[idx_c].push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue[idx_c].push_back(i_sram.RAM[cur_addr][7:0]);
							pre_zero_skip = i_sram.RAM[cur_addr][8];
							if(pre_zero_skip) $display($stime, " addr=%h have zero skip", cur_addr);
							cur_addr ++;
						end
					end
				end
			end

		SHT_RIGHT_UP: begin
				cur_addr = start_addr+1;
				for (int cur_h = 0; cur_h <= img_height_size; cur_h++) begin
					for (int cur_w = 0; cur_w <= img_width_size; cur_w++) begin
						if((cur_w == img_width_size) || (cur_h == 0)) begin
							 $display($stime, " push addr[%h] = right up zero", cur_addr);
							 check_queue[idx_c].push_back(0);
							 if((cur_w == img_width_size) && (cur_h != 0))
							 	cur_addr ++;
							 // if(pre_zero_skip == 1) begin
							 // 	cur_addr = cur_addr + 2;
							 // end else begin
							 // 	cur_addr = cur_addr + 1;
							 // end
							 // pre_zero_skip = 0;
						end else if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue[idx_c].push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue[idx_c].push_back(i_sram.RAM[cur_addr][7:0]);
							pre_zero_skip = i_sram.RAM[cur_addr][8];
							if(pre_zero_skip) $display($stime, " addr=%h have zero skip", cur_addr);
							cur_addr ++;
						end
					end
				end
			end

		SHT_RIGHT_DOWN: begin
				cur_addr = start_addr+img_width_size+2;
				for (int cur_h = 0; cur_h <= img_height_size; cur_h++) begin
					for (int cur_w = 0; cur_w <= img_width_size; cur_w++) begin
						if((cur_w == img_width_size) || (cur_h == img_height_size)) begin
							 $display($stime, " push addr[%h] = right down zero", cur_addr);
							 check_queue[idx_c].push_back(0);
							 if((cur_w == img_width_size))
							 	cur_addr ++;
							 // if(pre_zero_skip == 1) begin
							 // 	cur_addr = cur_addr + 2;
							 // end else begin
							 // 	cur_addr = cur_addr + 1;
							 // end
							 // pre_zero_skip = 0;
						end else if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue[idx_c].push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue[idx_c].push_back(i_sram.RAM[cur_addr][7:0]);
							pre_zero_skip = i_sram.RAM[cur_addr][8];
							if(pre_zero_skip) $display($stime, " addr=%h have zero skip", cur_addr);
							cur_addr ++;
						end
					end
				end
			end

		SHT_LEFT_UP: begin
				cur_addr = start_addr;
				for (int cur_h = 0; cur_h <= img_height_size; cur_h++) begin
					for (int cur_w = 0; cur_w <= img_width_size; cur_w++) begin
						if((cur_w == 0) || (cur_h == 0)) begin
							 $display($stime, " push addr[%h] = left up zero", cur_addr);
							 check_queue[idx_c].push_back(0);
							 if((cur_w == 0) && ~((cur_h == 0)||(cur_h == 1))) begin
							 	cur_addr = cur_addr + 1;
							 end
							 // if(pre_zero_skip == 1) begin
							 // 	cur_addr = cur_addr + 2;
							 // end else begin
							 // 	cur_addr = cur_addr + 1;
							 // end
							 // pre_zero_skip = 0;
						end else if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue[idx_c].push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue[idx_c].push_back(i_sram.RAM[cur_addr][7:0]);
							pre_zero_skip = i_sram.RAM[cur_addr][8];
							if(pre_zero_skip) $display($stime, " addr=%h have zero skip", cur_addr);
							cur_addr ++;
						end
					end
				end
			end

		SHT_LEFT_DOWN: begin
				cur_addr = start_addr + img_width_size + 1;
				for (int cur_h = 0; cur_h <= img_height_size; cur_h++) begin
					for (int cur_w = 0; cur_w <= img_width_size; cur_w++) begin
						if((cur_w == 0) || (cur_h == img_height_size)) begin
							 $display($stime, " push addr[%h] = left down zero", cur_addr);
							 check_queue[idx_c].push_back(0);
							 if((cur_w == 0) && (cur_h != 0)) begin
							 	cur_addr = cur_addr + 1;
							 end
							 // if(pre_zero_skip == 1) begin
							 // 	cur_addr = cur_addr + 2;
							 // end else begin
							 // 	cur_addr = cur_addr + 1;
							 // end
							 // pre_zero_skip = 0;
						end else if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue[idx_c].push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue[idx_c].push_back(i_sram.RAM[cur_addr][7:0]);
							pre_zero_skip = i_sram.RAM[cur_addr][8];
							if(pre_zero_skip) $display($stime, " addr=%h have zero skip", cur_addr);
							cur_addr ++;
						end
					end
				end
			end
		endcase // shift_ctrl
	endtask


	initial begin
		int cur_idx;
		int cur_w;
		int cur_h;
		int cur_addr;
		logic pre_zero_skip;
		repeat(5) get_clk;
		
		start_addr = 0;
		img_width_size  = IMG_W-1;
		img_height_size = IMG_H-1;


		`define FORK_RESULT_Q(idx_c) \
		fork \
			begin \
				forever begin \
					automatic logic [7:0]  check_result = 0; \
					get_result(.result_o(check_result), .idx(idx_c)); \
					$display($stime, " get   res[%d] <= ox%h", idx_c, check_result); \
					result_queue[idx_c].push_back(check_result); \
				end \
			end \
		join_none

		`FORK_RESULT_Q(0)
		`FORK_RESULT_Q(1)
		`FORK_RESULT_Q(2)
		`FORK_RESULT_Q(3)
		`FORK_RESULT_Q(4)
		`FORK_RESULT_Q(5)
		`FORK_RESULT_Q(6)
		`FORK_RESULT_Q(7)

		`define FORK_CHECK_Q(idx_c) \
		fork \
			begin \
				forever begin \
					int cur_idx; \
					cur_idx = 0; \
					if((result_queue[idx_c].size() != 0) && (check_queue[idx_c].size() != 0)) begin \
						automatic logic [7:0] pop_result; \
						automatic logic [7:0] pop_check; \
						pop_result = result_queue[idx_c].pop_front(); \
						pop_check  = check_queue[idx_c] .pop_front(); \
						$display($stime, " pop check=0x%h, result=0x%h for [%d][%d]", pop_check, pop_result, idx_c, cur_idx); \
						check_8(pop_check, pop_result); \
					end else if((result_queue[idx_c].size() != 0) && (check_queue[idx_c].size() == 0)) begin \
						$error($stime, " Error get more result @channel %d", idx_c); \
						$finish; \
					end else begin \
						get_clk; \
					end \
				end \
			end \
		join_none

		`FORK_CHECK_Q(0)
		`FORK_CHECK_Q(1)
		`FORK_CHECK_Q(2)
		`FORK_CHECK_Q(3)
		`FORK_CHECK_Q(4)
		`FORK_CHECK_Q(5)
		`FORK_CHECK_Q(6)
		`FORK_CHECK_Q(7)

		for (int j = 0; j <= img_height_size; j++) begin
			for (int i = 0; i <= img_width_size; i++) begin
				logic [7:0] x;
				x = j*(img_width_size+1) + i;
				if((j==0) || (i==0) || (i==img_width_size))            i_sram.RAM[start_addr+j*(img_width_size+1)+i][8] = 1'b0;
				else                                                   i_sram.RAM[start_addr+j*(img_width_size+1)+i][8] = 1'b0;
				i_sram.RAM[start_addr+j*(img_width_size+1)+i][7:0]     = x;
			end
		end



		// for (int ctrl = SHT_KEEP; ctrl <=SHT_LEFT_UP ; ctrl++) begin
		for (int idx_c=0; idx_c<8; idx_c++) begin
			int ctrl;
			std::randomize (ctrl) with  {(ctrl >= SHT_KEEP) && (ctrl <= SHT_LEFT_UP);};
			shift_ctrl[idx_c*4 +: 4] = ctrl;
			$display($stime, " Generate check for channel %d, ctrl=%d", idx_c, ctrl);
			generate_check(.ctrl(ctrl), .start_addr(start_addr), .idx_c(idx_c));
			$display($stime, "===============================================");
		end

		for (int idx_c=0; idx_c<8; idx_c++) begin
			$display($stime, " check_queue %d size = %d", idx_c, check_queue[idx_c].size());
		end




		get_clk;
		#1;
		shift_start = 1;
		get_clk;
		#1;
		shift_start = 0;

		repeat(2) get_clk;
		while(~shift_idle) get_clk;

		repeat(500) get_clk;

		foreach(check_queue[idx_c]) begin
			if(check_queue[idx_c].size()!=0) begin
				$error($stime, " Error check queue[%d] is not empty", idx_c);
				$finish;
			end
		end

		$finish;
	end
endmodule