module sram #(parameter DATA_WIDTH = 12, ADDR_WIDTH = 11, RAM_SIZE = 2048) (
  input                       clk   ,
  input                       we    ,
  input                       en    ,
  input      [ADDR_WIDTH-1:0] addr  ,
  input      [DATA_WIDTH-1:0] data_i,
  output reg [DATA_WIDTH-1:0] data_o
);

// Declareation of the memory cells
reg [DATA_WIDTH-1 : 0] RAM [RAM_SIZE - 1:0];

// ------------------------------------
// SRAM cell initialization
// ------------------------------------
// Initialize the sram cells with the values defined in "image.dat."
// initial begin
//     $readmemh("signals.mem", RAM);
// end

// ------------------------------------
// SRAM read operation
// ------------------------------------
always@(posedge clk)
begin
  if(en & ~we)
    data_o <= RAM[addr];
  else 
    data_o <= {(DATA_WIDTH){1'bx}};
end

// ------------------------------------
// SRAM write operation
// ------------------------------------
always@(posedge clk)
begin
  if (en & we)
    RAM[addr] <= data_i;
end

endmodule

