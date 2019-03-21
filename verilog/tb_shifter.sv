module tb_shifter;

	parameter IMG_W = 10;
	parameter IMG_H = 10;
	parameter SRAM_DEPTH = 32*32;

function [31:0] clog2 (input [31:0] x);
    reg [31:0] x_tmp;
    begin
            x_tmp = x-1;
            for(clog2=0; x_tmp>0; clog2=clog2+1) begin
                x_tmp = x_tmp >> 1;
            end
    end
endfunction

	logic                         clk            ='b0;
	logic                         reset_n        ='b0;
	logic [                  8:0] sram_data      ='b0;
	logic                         shift_start    ='b0;
	logic [                  3:0] shift_ctrl     ='b0;
	logic [clog2(SRAM_DEPTH)-1:0] start_addr     ='b0;
	logic [clog2(SRAM_DEPTH)-1:0] img_width_size ='b0;
	logic [clog2(SRAM_DEPTH)-1:0] img_height_size='b0;

	wire                         sram_en  ;
	wire [clog2(SRAM_DEPTH)-1:0] sram_addr;

	wire                          shift_idle     ;

	wire                          serial_output  ;
	wire                          serial_en      ;

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

	j_shifter #(.SRAM_DEPTH(SRAM_DEPTH)) i_j_shifter (
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

	logic [7:0] result_queue[$];
	logic [7:0] check_queue [$];

	// serial part
	task automatic get_clk();
		@(posedge clk);
	endtask : get_clk

	task automatic get_result_bit(output logic result_o);
		@(posedge clk);
		result_o = serial_output;
	endtask : get_result_bit

	task automatic get_result(output logic [7:0] result_o);
		for (int i = 0; i < 8; i = ((serial_en==1) ? i+1 : i)) begin
			get_result_bit(.result_o(result_o[i]));
		end
	endtask : get_result

	task automatic check_8(input logic [7:0] golden, input logic [7:0] check_result, input string err_msg="");
		if(golden !== check_result) begin
			$error($stime, " Error golden 0x%h != got 0x%h", golden, check_result);
			$finish;
		end
	endtask : check_8

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

		fork 
			begin 
				forever begin 
					automatic logic [7:0]  check_result = 0; 
					get_result(check_result); 
					$display($stime, " get   res <= ox%h", check_result); 
					result_queue.push_back(check_result); 
				end 
			end 
		join_none 

		fork
			begin
				forever begin
					int cur_idx;
					cur_idx = 0;
					if((result_queue.size() != 0) && (check_queue.size() != 0)) begin
						automatic logic [7:0] pop_result;
						automatic logic [7:0] pop_check;
						pop_result = result_queue.pop_front();
						pop_check  = check_queue .pop_front();
						$display($stime, " pop check=0x%h, result=0x%h for [%d]", pop_check, pop_result, cur_idx);
						check_8(pop_check, pop_result);
					end else if((result_queue.size() != 0) && (check_queue.size() == 0)) begin
						$error($stime, " Error get more result");
						$finish;
					end else begin
						get_clk;
					end
				end
			end
		join_none


		for (int j = 0; j <= img_height_size; j++) begin
			for (int i = 0; i <= img_width_size; i++) begin
				logic [7:0] x;
				x = j*(img_width_size+1) + i;
				if((j==0) || (i==0) || (i==img_width_size))            i_sram.RAM[start_addr+j*(img_width_size+1)+i][8] = 1'b0;
				else                                                   i_sram.RAM[start_addr+j*(img_width_size+1)+i][8] = 1'b0;
				i_sram.RAM[start_addr+j*(img_width_size+1)+i][7:0]     = x;
			end
		end


		for (int ctrl = SHT_KEEP; ctrl <=SHT_LEFT_UP ; ctrl++) begin
		
		shift_ctrl = ctrl;
		cur_idx = 0;
		pre_zero_skip = 0;
		case(shift_ctrl)
			SHT_KEEP: begin
				cur_addr = start_addr;
				for (int cur_h = 0; cur_h <= img_height_size; cur_h++) begin
					for (int cur_w = 0; cur_w <= img_width_size; cur_w++) begin
						if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue.push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue.push_back(i_sram.RAM[cur_addr][7:0]);
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
							check_queue.push_back(0);
						end else if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue.push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue.push_back(i_sram.RAM[cur_addr][7:0]);
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
							check_queue.push_back(0);
						end else if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue.push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue.push_back(i_sram.RAM[cur_addr][7:0]);
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
							check_queue.push_back(0);
						end else if(cur_w == 0) begin
							 $display($stime, " push addr[%h] = left zero", cur_addr);
							 check_queue.push_back(0);
							 // if(pre_zero_skip == 1) begin
							 // 	cur_addr = cur_addr + 2;
							 // end else begin
							 // 	cur_addr = cur_addr + 1;
							 // end
							 // pre_zero_skip = 0;
						end else if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue.push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue.push_back(i_sram.RAM[cur_addr][7:0]);
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
							 check_queue.push_back(0);
							 cur_addr ++;
							 // if(pre_zero_skip == 1) begin
							 // 	cur_addr = cur_addr + 2;
							 // end else begin
							 // 	cur_addr = cur_addr + 1;
							 // end
							 // pre_zero_skip = 0;
						end else if(pre_zero_skip == 1'b1) begin
							$display($stime, " push addr[%h] = zero", cur_addr);
							check_queue.push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue.push_back(i_sram.RAM[cur_addr][7:0]);
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
							 check_queue.push_back(0);
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
							check_queue.push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue.push_back(i_sram.RAM[cur_addr][7:0]);
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
							 check_queue.push_back(0);
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
							check_queue.push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue.push_back(i_sram.RAM[cur_addr][7:0]);
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
							 check_queue.push_back(0);
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
							check_queue.push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue.push_back(i_sram.RAM[cur_addr][7:0]);
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
							 check_queue.push_back(0);
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
							check_queue.push_back(0);
							pre_zero_skip = 1'b0;
						end else begin
							$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
							check_queue.push_back(i_sram.RAM[cur_addr][7:0]);
							pre_zero_skip = i_sram.RAM[cur_addr][8];
							if(pre_zero_skip) $display($stime, " addr=%h have zero skip", cur_addr);
							cur_addr ++;
						end
					end
				end
			end


		endcase // shift_ctrl

		get_clk;
		#1;
		shift_start = 1;
		get_clk;
		#1;
		shift_start = 0;

		while(~shift_idle) get_clk;

	    end // ctrl 




		repeat(500) get_clk;
		$finish;
	end
endmodule