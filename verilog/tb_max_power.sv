module tb_max_power;

	logic       clk                                ;
	logic       reset                              ;

	reg result_xor;
	reg result_en_xor;


	initial begin
		clk = 0;
		forever #5 clk = ~clk;
	end

	initial begin
		reset = 1'b0;
		@(posedge clk);
		reset = 1'b1;
		@(posedge clk);
		reset = 1'b0;
	end

	initial begin
	  $dumpfile("test.vcd");
	  $dumpvars;

	  repeat(4096) @(posedge clk);
	  $finish;
	end

	j_max_power_wrapper #(.SUBARRAY_WIDTH(32), .SUBARRAY_HEIGHT(32), .NUM_DATAFLOW_PER_MX(8)) i_j_max_power_wrapper (
		.clk          (clk          ),
		.reset        (reset        ),
		.result_xor   (result_xor   ),
		.result_en_xor(result_en_xor)
	);


endmodule