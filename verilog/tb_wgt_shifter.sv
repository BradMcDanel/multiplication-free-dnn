module tb_wgt_shifter;

	parameter IMG_SIZE    = 10   ;
	parameter SHIFT_WIDTH = 1    ;
	parameter SRAM_DEPTH  = 32*32;

function [31:0] clog2 (input [31:0] x);
    reg [31:0] x_tmp;
    begin
            x_tmp = x-1;
            for(clog2=0; x_tmp>0; clog2=clog2+1) begin
                x_tmp = x_tmp >> 1;
            end
    end
endfunction

	logic                         clk         = 'b0;
	logic                         reset_n     = 'b0;
	logic [                  7:0] sram_data   = 'b0;
	logic                         shift_start = 'b0;
	logic [clog2(SRAM_DEPTH)-1:0] end_addr    = 'b0;
	logic [clog2(SRAM_DEPTH)-1:0] img_size    = 'b0;

	wire                         sram_en  ;
	wire [clog2(SRAM_DEPTH)-1:0] sram_addr;

	wire                   shift_idle   ;
	wire [SHIFT_WIDTH-1:0] serial_output;
	wire                   serial_start ;
	wire                   serial_en    ;

	logic                         we    ;
	logic [clog2(SRAM_DEPTH)-1:0] addr  ;
	logic [                  7:0] data_i;
	wire  [                  7:0] data_o;

	sram #(.DATA_WIDTH(8), .ADDR_WIDTH(clog2(SRAM_DEPTH)), .RAM_SIZE(SRAM_DEPTH)) i_sram (
		.clk   (clk      ),
		.we    (1'b0     ),
		.en    (sram_en  ),
		.addr  (sram_addr),
		.data_i(9'b0     ),
		.data_o(sram_data)
	);

	j_wgt_shifter #(.SRAM_DEPTH(SRAM_DEPTH), .SHIFT_WIDTH(SHIFT_WIDTH)) i_j_wgt_shifter (
		.clk          (clk          ),
		.reset_n      (reset_n      ),
		.sram_en      (sram_en      ),
		.sram_addr    (sram_addr    ),
		.sram_data    (sram_data    ),
		.shift_start  (shift_start  ),
		.shift_idle   (shift_idle   ),
		.end_addr     (end_addr     ),
		.img_size     (img_size     ),
		.serial_output(serial_output),
		.serial_start (serial_start ),
		.serial_en    (serial_en    )
	);

	initial begin
	  $dumpfile("test.vcd");
	  $dumpvars;
	end

	initial begin
		clk = 0;
		forever #5 clk = ~clk;
	end

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

	task automatic get_result_bit(output logic [SHIFT_WIDTH-1:0] result_o);
		@(posedge clk);
		result_o = serial_output;
	endtask : get_result_bit

	task automatic get_result(output logic [7:0] result_o);
		for (int i = 0; i < (8/SHIFT_WIDTH); i = ((serial_en==1) ? i+1 : i)) begin
			get_result_bit(.result_o(result_o[i*SHIFT_WIDTH +: SHIFT_WIDTH]));
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
		int cur_addr;
		repeat(5) get_clk;
		
		end_addr  = IMG_SIZE-1;
		img_size  = IMG_SIZE-1;

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


		for (int i = 0; i <= img_size; i++) begin
			logic [31:0] x;
			x = i;
			i_sram.RAM[i][7:0]     = x;
		end

		
		cur_idx = 0;
		cur_addr = end_addr;
		for (int cur_idx = 0; cur_idx <= img_size; cur_idx++) begin
			$display($stime, " push addr[%h] = %h", cur_addr, i_sram.RAM[cur_addr][7:0]);
			check_queue.push_back(i_sram.RAM[cur_addr][31:0]);
			cur_addr --;
		end

		get_clk;
		#1;
		shift_start = 1;
		get_clk;
		#1;
		shift_start = 0;

		while(~shift_idle) get_clk;

		repeat(500) get_clk;
		$finish;
	end
endmodule